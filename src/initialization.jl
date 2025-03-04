function initialize_cubic!(x, y, z, npart, rc, halfdist)
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

function initialize_random(unitcell, npart, rng; tol=1.0)
    coordinates = unitcell[1] * rand(rng, StaticArrays.SVector{3,Float64}, npart)
    pack_monoatomic!(coordinates, unitcell, tol; parallel=false, iprint=100)

    return coordinates
end

function init_system(
    boxl, cutoff, inter_distance, rng, pathname; random=true, n_particles=10^3
)
    unitcell = [boxl, boxl, boxl]

    if random
        positions = initialize_random(unitcell, n_particles, rng; tol=1.2)
        # Save the initial configuration to a file
        open(joinpath(pathname, "initial.xyz"), "w") do io
            Printf.@printf(io, "%d\n", n_particles)
            # Print size of box
            Printf.@printf(
                io,
                "Lattice=\"%lf 0.0 0.0 0.0 %lf 0.0 0.0 0.0 %lf\"\n",
                unitcell[1],
                unitcell[2],
                unitcell[3],
            )
            # Print coordinates
            for i in eachindex(positions)
                particle = positions[i]
                Printf.@printf(
                    io, "%d %d %lf %lf %lf\n", 1, i, particle[1], particle[2], particle[3]
                )
            end
        end
    else
        # We can create normal arrays for holding the particles' positions
        x = zeros(n_particles)
        y = zeros(n_particles)
        z = zeros(n_particles)
        initialize_cubic!(x, y, z, n_particles, boxl / 2.0, inter_distance / 2.0)

        # Now we change the arrays to static versions of it
        positions = [@SVector [i, j, k] for (i, j, k) in zip(x, y, z)]
    end

    # Initialize system
    system = ParticleSystem(;
        xpositions=positions,
        unitcell=[boxl, boxl, boxl],
        cutoff=cutoff,
        output=EnergyAndForces(0.0, 0.0, similar(positions)),
        output_name=:energy_and_forces,
        parallel=false,
    )

    return system
end

function initialize_velocities(positions, ktemp, nf, rng, n_particles, dimension)
    # Initilize the random numbers of the velocities
    velocities = [StaticArrays.@SVector zeros(dimension) for _ in 1:length(positions)]
    sum_v = StaticArrays.@MVector zeros(dimension)
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
