function initialize_random(unitcell, npart, rng, dimension; tol=1.0)
    coordinates = unitcell[1] * rand(rng, StaticArrays.SVector{dimension,Float64}, npart)
    pack_monoatomic!(coordinates, unitcell, tol; parallel=false, iprint=100)

    return coordinates
end

function init_system(
    boxl,
    cutoff,
    inter_distance,
    rng,
    pathname,
    dimension,
    diameters;
    random=true,
    n_particles=10^3,
)
    unitcell = boxl .* ones(dimension)

    if random
        positions = initialize_random(unitcell, n_particles, rng, dimension; tol=1.1)
        # Save the initial configuration to a file
        write_to_file(
            joinpath(pathname, "packed.xyz"),
            0,
            boxl,
            n_particles,
            positions,
            diameters,
            dimension;
            mode="w",
        )
    else
        # ! This has to change depending on the dimension
        # We can create normal arrays for holding the particles' positions
        x = zeros(n_particles)
        y = zeros(n_particles)
        z = zeros(n_particles)
        # ! FIXME: This is now broken, waiting for SimpleCrystals.jl implementation
        initialize_cubic!(x, y, z, n_particles, boxl / 2.0, inter_distance / 2.0)

        # Now we change the arrays to static versions of it
        positions = [@SVector [i, j, k] for (i, j, k) in zip(x, y, z)]
    end

    # Initialize system
    system = ParticleSystem(;
        xpositions=positions,
        unitcell=unitcell,
        cutoff=cutoff,
        output=EnergyAndForces(0.0, 0.0, similar(positions)),
        output_name=:energy_and_forces,
        parallel=false,
    )

    return system
end

function initialize_simulation(
    params::Parameters, rng, dimension, diameters, pathname; file="", random_init=false
)
    # Always leave a fixed cutoff
    cutoff = 1.5
    positions = []
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
            parallel=false,
        )

        # Save the initial configuration to a file
        filepath = joinpath(pathname, "initial.xyz")
        write_to_file(
            filepath, 0, boxl, params.n_particles, positions, diameters, dimension
        )

        @info "Reading done."
    else
        @info "Creating a new system with random positions and no overlaps."
        # Now we compute the effective size of the box
        inter_distance = (1.0 / params.ρ)^(1 / dimension)
        boxl = (params.n_particles / params.ρ)^(1 / dimension)

        # Initialize the system in a lattice configuration
        system = init_system(
            boxl,
            cutoff,
            inter_distance,
            rng,
            pathname,
            dimension,
            diameters;
            random=random_init,
            n_particles=params.n_particles,
        )
    end

    return system, boxl
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
    reset_output!(system.energy_and_forces)

    # Initialize the array of images
    images = [StaticArrays.@MVector zeros(Int32, dimension) for _ in eachindex(velocities)]

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
