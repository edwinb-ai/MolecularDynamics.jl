using Random
using StaticArrays
using LinearAlgebra
using DelimitedFiles
using Printf

struct Parameters
    ρ::Float64
    ktemp::Float64
    n_particles::Int
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

    if rij < b_param
        uij = (a_param / ktemp) * ((1.0 / rij)^lambda - (1.0 / rij)^(lambda - 1.0))
        uij += (1.0 / ktemp)
        fij = lambda * (1.0 / rij)^(lambda + 1.0)
        fij -= (lambda - 1.0) * (1.0 / rij)^lambda
        fij *= -a_param / ktemp
    else
        uij = 0.0
        fij = 0.0
    end

    return uij, fij
end

function lj(rij, sigma=1.0)
    uij = 4.0 * ((sigma / rij)^12 - (sigma / rij)^6)
    fij = 24.0 * (2.0 * (sigma^12 / rij^12) - (sigma^6 / rij^6))

    return uij, fij
end

function lrc(cutoff, density, sigma=1.0)
    uij = (((sigma / cutoff)^9) / 9.0) - (((sigma / cutoff)^3) / 3.0)
    uij *= 8.0 * pi * density

    return uij
end

function energy_force!(positions, forces, boxl, cutoff, n_particles)
    total_energy = 0.0
    virial = 0.0

    # Zero out the force arrays
    for i in eachindex(forces)
        forces[i] = @SVector zeros(3)
    end

    for i in 1:(n_particles - 1)
        for j in (i + 1):n_particles
            uij = 0.0
            fij = 0.0
            # Pair contribution
            rij = positions[i] - positions[j]
            # Periodic boundary conditions
            rij = @. rij - boxl * round(rij / boxl)
            distance = norm(rij)
            if distance < cutoff
                uij, fij = lj(distance)
                total_energy += uij
                sumies = @. fij * rij / distance
                virial += dot(sumies, rij)
            end
            # Update the forces between particles
            forces[i] = @. forces[i] + (fij * rij / distance)
            forces[j] = @. forces[j] - (fij * rij / distance)
        end
    end

    return total_energy, virial
end

function integrate_half(positions, velocities, forces, dt, boxl; pbc=true)
    # ! Important: There is a mass in the force term
    new_velocities = @. velocities + (forces * dt / 2.0)
    new_positions = @. positions + (new_velocities * dt)
    # Periodic boundary conditions
    if pbc
        new_positions = @. new_positions - boxl * round(new_positions / boxl)
    end

    return new_positions
end

function andersen!(velocities, ktemp, const_val, rng)
    sigma = sqrt(ktemp)
    noise = @MVector zeros(3)

    for i in eachindex(velocities)
        if rand(rng) < const_val
            randn!(rng, noise)
            velocities[i] = noise .* sigma
        end
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
    rng = Random.Xoshiro(123)
    boxl = cbrt(params.n_particles / params.ρ)
    inter_distance = cbrt(1.0 / params.ρ)
    cutoff = 3.0
    dt = 0.005
    nu_const = 1.0 * dt
    # The degrees of freedom
    # Spatial dimension, in this case 3D simulations
    dimension = 3.0
    nf = dimension * (params.n_particles - 1.0)
    ener_lrc = lrc(cutoff, params.ρ)
    kinetic_energy = 0.0

    # We can create normal arrays for holding the particles' positions
    x = zeros(params.n_particles)
    y = zeros(params.n_particles)
    z = zeros(params.n_particles)
    initialize_simulation!(x, y, z, params.n_particles, boxl / 2.0, inter_distance / 2.0)
    # Now we change the arrays to static versions of it
    positions = [@SVector [i, j, k] for (i, j, k) in zip(x, y, z)]

    # Initialize the velocities of the system by having the correct temperature
    velocities = initialize_velocities(positions, params.ktemp, nf, rng, params.n_particles)
    # Adjust the particles using the velocities
    for i in eachindex(positions)
        positions[i] = @. positions[i] - (velocities[i] * dt)
    end
    # Save the initial configuration for easier visualization
    open(joinpath(pathname, "inital_configuration.xyz"), "w") do io
        println(io, params.n_particles)
        println(io, "Frame")
        writedlm(io, [x y z], " ")
    end

    # We also need the array for the forces
    forces = [@SVector zeros(3) for _ in 1:(params.n_particles)]
    ener, _ = energy_force!(positions, forces, boxl, cutoff, params.n_particles)

    # Open files and start accumulators
    trajectory = open(joinpath(pathname, "trajectory.xyz"), "w")
    energy_file = open(joinpath(pathname, "energy.dat"), "w")
    zfactor = 0.0
    nprom = 0

    # Start the simulation
    for t in 1:(eq_steps + prod_steps)
        # First half of the integration
        for i in eachindex(positions, forces, velocities)
            x = positions[i]
            f = forces[i]
            v = velocities[i]
            new_x = integrate_half(x, v, f, dt, boxl)
            velocities[i] = @. v + (f * dt / 2.0)
            # Always assign the positions to the already existing array
            positions[i] = new_x
        end

        # Compute the energy and the forces
        ener, vir = energy_force!(positions, forces, boxl, cutoff, params.n_particles)

        # Second half of the integration
        for i in eachindex(forces, velocities)
            f = forces[i]
            velocities[i] = @. velocities[i] + (f * dt / 2.0)
        end

        # Apply the Andersen thermostat
        andersen!(velocities, params.ktemp, nu_const, rng)
        kinetic_energy = 0.0
        for i in eachindex(velocities)
            kinetic_energy += sum(abs2, velocities[i])
        end
        kinetic_energy /= 2.0

        temperature = 2.0 * kinetic_energy / nf
        ener /= params.n_particles
        ener += ener_lrc
        @show ener, temperature

        if mod(t, 10) == 0 && t > eq_steps
            zfactor += vir
            nprom += 1
        end

        # Every few steps we show the energy and save to disk
        if mod(t, 100) == 0 && t > eq_steps
            temperature = 2.0 * kinetic_energy / nf
            ener /= params.n_particles
            ener += ener_lrc
            pressure = 1.0 - (zfactor / (3.0 * nprom * params.n_particles * params.ktemp))
            writedlm(energy_file, [pressure temperature ener], " ")
        end

        # Save to disk the positions
        if mod(t, 10) == 0 && t > eq_steps
            # Write to file
            println(trajectory, params.n_particles)
            println(trajectory, "Frame $t")
            for p in positions
                writedlm(trajectory, [p], " ")
            end
        end
    end

    # Close down all files
    close(trajectory)
    close(energy_file)

    return nothing
end

function main()
    # packings = [0.45]
    # densities = phi2density.(packings)
    densities = [0.9]

    for d in densities
        params = Parameters(d, 0.85, 5^3)
        # Create a new directory with these parameters
        pathname = joinpath(@__DIR__, "packing=$(@sprintf("%.3f", d))")
        mkpath(pathname)
        simulation(params, pathname; eq_steps=100_000, prod_steps=1_000_000)
    end

    return nothing
end

main()
