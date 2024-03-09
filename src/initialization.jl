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

function initialize_velocities(positions, ktemp, nf, rng, n_particles)
    # Initilize the random numbers of the velocities
    velocities = [StaticArrays.@SVector zeros(3) for _ in 1:length(positions)]
    sum_v = StaticArrays.@MVector zeros(3)
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
