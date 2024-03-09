using Random
using StaticArrays
using LinearAlgebra
using DelimitedFiles
using Printf
using FastPow
using ThreadPools
using Distributions: Gamma
using CellListMap.PeriodicSystems
import CellListMap.PeriodicSystems: copy_output, reset_output!, reducer

struct Parameters
    ρ::Float64
    ktemp::Float64
    n_particles::Int
end

mutable struct EnergyAndForces
    energy::Float64
    virial::Float64
    forces::Vector{SVector{3,Float64}}
end

function initialize_simulation!(x, y, z, npart, rc, halfdist)
    dist = halfdist * 2.0

    # Define the first positions
    x[1] = -rc + halfdist
    y[1] = -rc + halfdist
    z[1] = -rc + halfdist

    # Create a complete lattice
    for i in 1:(npart - 1)
        x[i + 1] = x[i] + dist
        y[i + 1] = y[i]
        z[i + 1] = z[i]

        if x[i + 1] > rc
            x[i + 1] = -rc + halfdist
            y[i + 1] += dist
            z[i + 1] = z[i]

            if y[i + 1] > rc
                x[i + 1] = -rc + halfdist
                y[i + 1] = -rc + halfdist
                z[i + 1] += dist
            end
        end
    end

    return nothing
end

function pseudohs(rij; lambda=50.0)
    b_param = lambda / (lambda - 1.0)
    a_param = lambda * b_param^(lambda - 1.0)
    ktemp = 1.0
    uij = 0.0
    fij = 0.0

    if rij < b_param
        uij = (a_param / ktemp) * ((1.0 / rij)^lambda - (1.0 / rij)^(lambda - 1.0))
        uij += (1.0 / ktemp)
        fij = lambda * (1.0 / rij)^(lambda + 1.0)
        fij -= (lambda - 1.0) * (1.0 / rij)^lambda
        fij *= -a_param / ktemp
    end

    return uij, fij
end

FastPow.@fastpow function lj(rij, sigma=1.0)
    # pot_cut = (sigma / cutoff)^12 - (sigma / cutoff)^6
    uij = (sigma / rij)^12 - (sigma / rij)^6
    # uij -= pot_cut
    uij *= 4.0
    fij = 24.0 * (2.0 * (sigma / rij)^13 - (sigma / rij)^7)

    return uij, fij
end

function ener_lrc(cutoff, density, sigma=1.0)
    uij = (((sigma / cutoff)^9) / 3.0) - ((sigma / cutoff)^3)
    uij *= 8.0 * pi * density / 3.0

    return uij
end

function pressure_lrc(cutoff, density, sigma=1.0)
    sr3 = (sigma / cutoff)^3
    result = (2.0 * sr3^3 / 3.0) - sr3
    result *= 16.0 * pi * density^2 / 3.0
    # delta = sr3^3 - sr3
    # delta *= 8.0 * pi * density^2 / 3.0

    return result
end

"Custom copy, reset and reducer functions"
function copy_output(x::EnergyAndForces)
    return EnergyAndForces(copy(x.energy), copy(x.virial), copy(x.forces))
end

function reset_output!(output::EnergyAndForces)
    output.energy = 0.0
    output.virial = 0.0

    for i in eachindex(output.forces)
        output.forces[i] = SVector(0.0, 0.0, 0.0)
    end

    return output
end

function reducer(x::EnergyAndForces, y::EnergyAndForces)
    e_tot = x.energy + y.energy
    vir_tot = x.virial + y.virial
    x.forces .+= y.forces

    return EnergyAndForces(e_tot, vir_tot, x.forces)
end

"Function that updates energy and forces for each pair"
function energy_and_forces!(x, y, i, j, d2, output::EnergyAndForces)
    d = sqrt(d2)
    r = x - y
    (uij, fij) = lj(d)
    sumies = @. fij * r / d
    output.virial += dot(sumies, r)
    # output.virial += virial / 2.0
    output.energy += uij
    output.forces[i] = @. output.forces[i] + sumies
    output.forces[j] = @. output.forces[j] - sumies

    return output
end

function init_system(boxl, cutoff, inter_distance; n_particles=10^3)
    # We can create normal arrays for holding the particles' positions
    x = zeros(n_particles)
    y = zeros(n_particles)
    z = zeros(n_particles)
    initialize_simulation!(x, y, z, n_particles, boxl / 2.0, inter_distance / 2.0)

    # Now we change the arrays to static versions of it
    positions = [@SVector [i, j, k] for (i, j, k) in zip(x, y, z)]
    # Initialize system
    system = PeriodicSystem(;
        xpositions=positions,
        unitcell=[boxl, boxl, boxl],
        cutoff=cutoff,
        output=EnergyAndForces(0.0, 0.0, similar(positions)),
        output_name=:energy_and_forces,
        parallel=false,
    )

    return system
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

function andersen!(velocities, ktemp, const_val, rng)
    sigma = sqrt(ktemp)

    for i in eachindex(velocities)
        if rand(rng) < const_val
            noise = @SVector randn(rng, 3)
            velocities[i] = noise .* sigma
        end
    end

    return nothing
end

function sum_noises(nf, rng)
    result = 0.0

    if nf == 0.0
        result = 0.0
    elseif nf == 1.0
        result = randn(rng)^2
    elseif mod(nf, 2) == 0
        gamma_dist = Gamma(nf ÷ 2)
        result = 2.0 * rand(rng, gamma_dist)
    else
        gamma_dist = Gamma((nf - 1) ÷ 2)
        result = 2.0 * rand(rng, gamma_dist)
        result += randn(rng)^2
    end

    return result
end

function bussi!(velocities, ktemp, nf, dt, τ, rng)
    dt_ratio = dt / τ

    # Compute kinetic energy
    kinetic_energy = 0.0
    for i in eachindex(velocities)
        kinetic_energy += sum(abs2, velocities[i])
    end
    kinetic_energy /= 2.0
    current_temperature = 2.0 * kinetic_energy / nf

    # Compute random numbers
    r1 = randn(rng)
    r2 = sum_noises(nf - 1, rng)

    # Compute the parameters from the thermostat
    term_1 = exp(-dt_ratio)
    c2 = (1.0 - term_1) * ktemp / (current_temperature * nf)
    term_2 = c2 * (r2 + r1^2)
    term_3 = 2.0 * r1 * sqrt(term_1 * c2)
    scale = sqrt(term_1 + term_2 + term_3)

    # Apply velocity rescaling
    for i in eachindex(velocities)
        velocities[i] = velocities[i] * scale
    end

    return nothing
end

function initialize_velocities(positions, ktemp, nf, rng, n_particles)
    # Initilize the random numbers of the velocities
    velocities = [@SVector zeros(3) for _ in 1:length(positions)]
    sum_v = @MVector zeros(3)
    sum_v2 = 0.0

    for i in eachindex(velocities)
        velocities[i] = randn(rng, size(velocities[i]))
        # Collect the center of mass, momentum = 1
        sum_v .+= velocities[i]
    end

    sum_v ./= n_particles

    for i in eachindex(velocities)
        # Remove the center of mass momentum
        velocities[i] = velocities[i] .- sum_v
        sum_v2 += sum(abs2, velocities[i])
    end

    fs = sqrt(ktemp / (sum_v2 / nf))
    for i in eachindex(velocities)
        velocities[i] = velocities[i] .* fs
    end

    return velocities
end

function simulation(params::Parameters, pathname; eq_steps=100_000, prod_steps=500_000)
    rng = Random.Xoshiro()
    boxl = cbrt(params.n_particles / params.ρ)
    volume = boxl^3
    inter_distance = cbrt(1.0 / params.ρ)
    cutoff = 3.0
    dt = 0.005
    # nu_const = 1.0 * dt
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

    # Initialize the system
    system = init_system(boxl, cutoff, inter_distance; n_particles=params.n_particles)
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

        # Apply the Andersen thermostat only during equilibration
        if step <= eq_steps
            # andersen!(velocities, params.ktemp, nu_const, rng)
            bussi!(velocities, params.ktemp, nf, dt, τ, rng)
        end
        kinetic_energy = 0.0
        for i in eachindex(velocities)
            kinetic_energy += sum(abs2, velocities[i])
        end
        kinetic_energy /= 2.0

        # Accumulate the values of the virial
        if mod(step, 100) == 0 && step > eq_steps
            virial += system.energy_and_forces.virial
            nprom += 1
        end

        # Every few steps we show the energy and save to disk
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
                writedlm(trajectory_file, [1 i system.positions[i]...], " ")
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
