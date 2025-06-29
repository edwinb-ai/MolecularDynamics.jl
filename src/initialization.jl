function initialize_random(unitcell, npart, rng, dimension; tol=1.0)
    coordinates = unitcell[1] * rand(rng, StaticArrays.SVector{dimension,Float64}, npart)
    pack_monoatomic!(coordinates, unitcell, tol; parallel=true, iprint=100)

    return coordinates
end

function initialize_simulation(
    params::Parameters,
    rng,
    dimension,
    pathname;
    cutoff::Float64=1.5,
    from_file::String="",
    random_init::Bool=false,
)
    boxl = 0.0
    system = nothing

    # We have to make sure that we only use either one, from a file or random initialization
    if isfile(from_file) || !random_init
        @info "Reading from file..."
        (boxl, positions, diameters) = read_file(from_file; dimension=dimension)

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
        @info "Reading done."
        # If either one is not satisfied, then we create a random initialization for now
    else
        @info "Creating a new system with random positions and no overlaps."
        # Now we compute the effective size of the box
        boxl = (params.n_particles / params.ρ)^(1 / dimension)
        unitcell = boxl .* ones(dimension)

        # FIXME: We set an array of diameters here, but the user should be able to pass
        # it as an argument
        diameters = ones(params.n_particles)

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

    return system, boxl, diameters
end

function initialize_velocities(ktemp, rng, n_particles, dimension)
    # 1) draw all velocities at once into a matrix
    V = randn(rng, dimension, n_particles)         # size: (d × N)
    # 2) remove COM motion
    V .-= mean(V; dims=2)                          # subtract column-wise mean
    # 3) compute current total squared speed
    sum_v2 = sum(abs2, V)
    # 4) compute scale factor
    fs = sqrt(ktemp / (sum_v2 / ((n_particles - 1) * dimension)))
    # 5) apply in place
    V .*= fs
    # 6) if you still need SVectors
    velocities = [StaticArrays.MVector{dimension,Float64}(V[:, i]) for i in 1:n_particles]

    return velocities
end

function initialize_state(
    params::Parameters,
    pathname::String;
    from_file::String="",
    dimension::Int=3,
    random_init=false,
    cutoff=1.5,
    rng::Random.AbstractRNG=Random.Xoshiro(),
)
    # The degrees of freedom
    nf = dimension * (params.n_particles - 1.0)

    # Initialize the system
    (system, boxl, diameters) = initialize_simulation(
        params,
        rng,
        dimension,
        pathname;
        cutoff=cutoff,
        from_file=from_file,
        random_init=random_init,
    )
    
    # Zero out the arrays
    for i in eachindex(system.energy_and_forces.forces)
        system.energy_and_forces.forces[i] = zeros(StaticArrays.SVector{dimension})
    end
    reset_output!(system.energy_and_forces)

    # Initialize the array of images
    images = [
        zeros(StaticArrays.MVector{dimension,Int32}) for _ in eachindex(system.xpositions)
    ]

    # We set the velocities to zero for now, they will be initialized later
    state = SimulationState(
        system,
        diameters,
        rng,
        boxl,
        Vector{StaticArrays.SVector{dimension,Float64}}(),
        images,
        dimension,
        nf,
    )

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
