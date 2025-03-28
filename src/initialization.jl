function initialize_random(unitcell, npart, rng, dimension; tol=1.0)
    coordinates = unitcell[1] * rand(rng, StaticArrays.SVector{dimension,Float64}, npart)
    pack_monoatomic!(coordinates, unitcell, tol; parallel=true, iprint=100)

    return coordinates
end

function initialize_simulation(
    params::Parameters, rng, dimension, diameters, pathname; file="", random_init=false
)
    # Always leave a fixed cutoff
    cutoff = 1.5
    boxl = 0.0
    system = nothing

    if isfile(file)
        @info "Reading from file..."
        (boxl, positions, diameters) = read_file(file)

        # Initialize system
        unitcell = boxl .* ones(dimension)
        system = CellListMap.ParticleSystem(;
            xpositions=positions,
            unitcell=unitcell,
            cutoff=cutoff,
            output=EnergyAndForces(0.0, 0.0, similar(positions)),
            output_name=:energy_and_forces,
            parallel=true,
        )

        # Save the initial configuration to a file
        filepath = joinpath(pathname, "initial.xyz")
        write_to_file(
            filepath, 0, boxl, params.n_particles, positions, diameters, dimension
        )

        @info "Reading done."
    elseif random_init
        @info "Creating a new system with random positions and no overlaps."
        # Now we compute the effective size of the box
        boxl = (params.n_particles / params.ρ)^(1 / dimension)
        unitcell = boxl .* ones(dimension)

        positions = initialize_random(unitcell, params.n_particles, rng, dimension)
        # Save the initial configuration to a file
        write_to_file(
            joinpath(pathname, "packed.xyz"),
            0,
            boxl,
            params.n_particles,
            positions,
            diameters,
            dimension;
            mode="w",
        )

        # Initialize system
        system = ParticleSystem(;
            xpositions=positions,
            unitcell=unitcell,
            cutoff=cutoff,
            output=EnergyAndForces(0.0, 0.0, similar(positions)),
            output_name=:energy_and_forces,
            parallel=true,
        )
    end

    return system, boxl
end

function initialize_velocities(positions, ktemp, nf, rng, n_particles, dimension)
    # Initilize the random numbers of the velocities
    velocities = [zeros(SVector{dimension,Float64}) for _ in 1:length(positions)]
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

function initialize_state(
    params::Parameters,
    ktemp::Float64,
    pathname::String,
    diameters::Vector{Float64};
    from_file::String="",
    dimension::Int=3,
    random_init=false,
)
    rng = Random.Xoshiro()

    # The degrees of freedom
    nf = dimension * (params.n_particles - 1.0)

    # Initialize the system
    (system, boxl) = initialize_simulation(
        params, rng, dimension, diameters, pathname; file=from_file, random_init=random_init
    )
    # Initialize the velocities of the system by having the correct temperature
    velocities = initialize_velocities(
        system.positions, ktemp, nf, rng, params.n_particles, dimension
    )
    # Adjust the particles using the velocities
    for i in eachindex(system.positions)
        system.positions[i] = @. system.positions[i] - (velocities[i] * params.dt)
    end
    # Zero out the arrays
    for i in eachindex(system.energy_and_forces.forces)
        system.energy_and_forces.forces[i] = zeros(StaticArrays.SVector{dimension})
    end
    reset_output!(system.energy_and_forces)

    # Initialize the array of images
    images = [zeros(StaticArrays.MVector{dimension,Int32}) for _ in eachindex(velocities)]

    state = SimulationState(system, diameters, rng, boxl, velocities, images, dimension, nf)

    # Let's write the initial configuration to a file
    write_to_file(
        joinpath(pathname, "init.xyz"),
        0,
        boxl,
        params.n_particles,
        system.positions,
        diameters,
        dimension;
        mode="w",
    )

    return state
end
