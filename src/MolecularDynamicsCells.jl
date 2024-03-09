using Random
using StaticArrays
using LinearAlgebra: dot
using DelimitedFiles: writedlm
using Printf
using FastPow
using ThreadPools
using Distributions: Gamma
using CellListMap.PeriodicSystems
import CellListMap.PeriodicSystems: copy_output, reset_output!, reducer
using Packmol: pack_monoatomic!

include("initialization.jl")
include("potentials.jl")
include("thermostat.jl")
include("pairwise.jl")

struct Parameters
    ρ::Float64
    ktemp::Float64
    n_particles::Int
end

function integrate_half(positions, velocities, forces, dt, boxl; pbc=true)
    # ! Important: There is a mass in the force term
    new_velocities = @. velocities + (forces * dt / 2.0)
    new_positions = @. positions + (new_velocities * dt)
    # Periodic boundary conditions
    if pbc
        new_positions = @. new_positions - boxl * round(new_positions / boxl)
    end

    return new_positions, new_velocities
end

function simulation(params::Parameters, pathname; eq_steps=100_000, prod_steps=500_000)
    rng = Random.Xoshiro()
    boxl = cbrt(params.n_particles / params.ρ)
    volume = boxl^3
    inter_distance = cbrt(1.0 / params.ρ)
    cutoff = 3.0
    dt = 0.005
    τ = 100.0 * dt
    # The degrees of freedom
    # Spatial dimension, in this case 3D simulations
    dimension = 3.0
    nf = dimension * (params.n_particles - 1.0)
    # Compute LRC for LJ
    energy_lrc = ener_lrc(cutoff, params.ρ)
    press_lrc = pressure_lrc(cutoff, params.ρ)

    # Variables to accumulate results
    virial = 0.0
    nprom = 0
    kinetic_energy = 0.0

    # Initialize the system in a random configuration
    system = init_system(
        boxl,
        cutoff,
        inter_distance,
        rng,
        pathname;
        random=true,
        n_particles=params.n_particles,
    )
    # Initialize the velocities of the system by having the correct temperature
    velocities = initialize_velocities(
        system.positions, params.ktemp, nf, rng, params.n_particles
    )
    # Adjust the particles using the velocities
    for i in eachindex(system.positions)
        system.positions[i] = @. system.positions[i] - (velocities[i] * dt)
    end
    # Zero out the arrays
    reset_output!(system.energy_and_forces)

    # Open files for trajectory and other things
    trajectory_file = open(joinpath(pathname, "production.xyz"), "w")
    thermo_file = open(joinpath(pathname, "thermo.txt"), "w")

    for step in 1:(eq_steps + prod_steps)
        # First half of the integration
        for i in eachindex(system.positions, system.energy_and_forces.forces, velocities)
            f = system.energy_and_forces.forces[i]
            x = system.positions[i]
            v = velocities[i]
            (new_x, new_v) = integrate_half(x, v, f, dt, boxl)
            velocities[i] = new_v
            system.positions[i] = new_x
        end

        # Zero out arrays
        reset_output!(system.energy_and_forces)
        # Compute energy and forces
        map_pairwise!(
            (x, y, i, j, d2, output) -> energy_and_forces!(x, y, i, j, d2, output), system
        )

        # Second half of the integration
        for i in eachindex(velocities, system.energy_and_forces.forces)
            f = system.energy_and_forces.forces[i]
            v = velocities[i]
            velocities[i] = @. v + (f * dt / 2.0)
        end

        # Always apply the thermostat and compute the kinetic energy
        bussi!(velocities, params.ktemp, nf, dt, τ, rng)
        kinetic_energy = 0.0
        for i in eachindex(velocities)
            kinetic_energy += sum(abs2, velocities[i])
        end
        kinetic_energy /= 2.0

        # Accumulate the values of the virial for computing the pressure
        if mod(step, 100) == 0 && step > eq_steps
            virial += system.energy_and_forces.virial
            nprom += 1
        end

        # Every few steps we save thermodynamic quantities to disk
        if mod(step, 100) == 0 && step > eq_steps
            ener_part = system.energy_and_forces.energy
            ener_part /= params.n_particles
            tot_eng = ener_part + energy_lrc
            temperature = 2.0 * kinetic_energy / nf
            pressure = virial / (dimension * nprom * volume)
            pressure += params.ρ * temperature
            pressure += press_lrc
            writedlm(thermo_file, [ener_part tot_eng temperature pressure], " ")
        end

        # Save to disk the positions
        if mod(step, 10000) == 0 && step > eq_steps
            # Write to file
            println(trajectory_file, params.n_particles)
            println(trajectory_file, "Frame $step")
            for i in eachindex(system.positions)
                particle = system.positions[i]
                Printf.@printf(
                    io, "%d %d %lf %lf %lf\n", 1, i, particle[1], particle[2], particle[3]
                )
            end
        end
    end

    # Close all opened files
    close(trajectory_file)
    close(thermo_file)

    return nothing
end

function main()
    densities = [0.776, 0.78, 0.82, 0.84, 0.86, 0.9]
    # densities = [0.9]

    ThreadPools.@qthreads for d in densities
        # for d in densities
        params = Parameters(d, 0.85, 8^3)
        # Create a new directory with these parameters
        pathname = joinpath(@__DIR__, "bussi-cells_density=$(@sprintf("%.4g", d))")
        mkpath(pathname)
        simulation(params, pathname; eq_steps=100_000, prod_steps=1_000_000)
    end

    return nothing
end

main()
