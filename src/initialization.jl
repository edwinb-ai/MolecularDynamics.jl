function initialize_random(unitcell, npart, rng, dimension; tol=1.0)
    coordinates = unitcell[1] * rand(rng, StaticArrays.SVector{dimension,Float64}, npart)
    pack_monoatomic!(coordinates, unitcell, tol; parallel=true, iprint=100)

    return coordinates
end

"""
    initialize_simulation(
        params::Parameters,
        rng,
        dimension,
        pathname;
        cutoff::Float64=1.5,
        from_file::String="",
        random_init::Bool=false,
        positions=nothing,
        diameters=nothing,
        box_type::Union{String, Symbol}="cubic",
        box_size=nothing,
    )

Flexible simulation system initialization.

- If `positions` and `diameters` are provided, they are used directly (manual mode); `box_type` and `box_size` determine the box.
- If `from_file` is given, loads system from file.
- Otherwise, generates random/packed configuration from `params` and `box_type`.
"""
function initialize_simulation(
    params::Parameters,
    rng,
    dimension,
    pathname;
    cutoff::Float64=1.5,
    from_file::String="",
    random_init::Bool=false,
    positions=nothing,
    diameters=nothing,
    box_type::Union{String,Symbol}="cubic",
    box_size=nothing,
)
    boxl = 0.0
    system = nothing

    if positions !== nothing && diameters !== nothing
        @info "Initializing from provided positions/diameters..."
        if isnothing(box_size)
            # Default box size: bounding box of positions plus a margin
            minpos = minimum(reduce(hcat, positions); dims=2)
            maxpos = maximum(reduce(hcat, positions); dims=2)
            margin = maximum(diameters)
            box_size = vec(maxpos .- minpos) .+ margin
        end
        unitcell = if (box_type == "cubic" || box_type == :cubic)
            box_size .* ones(dimension)
        else
            box_size
        end
        system = ParticleSystem(;
            xpositions=positions,
            unitcell=unitcell,
            cutoff=cutoff,
            output=EnergyAndForces(0.0, 0.0, similar(positions)),
            output_name=:energy_and_forces,
            parallel=true,
        )
        boxl = box_size
    elseif isfile(from_file) || !random_init
        @info "Reading from file..."
        (boxl, positions, diameters) = read_file(from_file; dimension=dimension)
        unitcell = boxl .* ones(dimension)
        system = ParticleSystem(;
            xpositions=positions,
            unitcell=unitcell,
            cutoff=cutoff,
            output=EnergyAndForces(0.0, 0.0, similar(positions)),
            output_name=:energy_and_forces,
            parallel=true,
        )
        @info "Reading done."
    else
        @info "Creating a new system with random positions and no overlaps."
        boxl = if box_size === nothing
            (params.n_particles / params.ρ)^(1 / dimension)
        else
            (typeof(box_size) <: Number ? box_size : maximum(box_size))
        end
        unitcell =
            (box_type == "cubic" || box_type == :cubic) ? boxl .* ones(dimension) : box_size
        diameters = diameters === nothing ? ones(params.n_particles) : diameters
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

"""
    initialize_state(
        params::Parameters,
        pathname::String;
        from_file::String="",
        dimension::Int=3,
        random_init=false,
        cutoff=1.5,
        rng::Random.AbstractRNG=Random.Xoshiro(),
        positions=nothing,
        diameters=nothing,
        box_type::Union{String, Symbol}="cubic",
        box_size=nothing,
    )

Flexible state initialization.

- If `positions` and `diameters` are supplied, they take precedence and the box is inferred from `box_type` and `box_size`.
- If `from_file` is supplied, loads state from file.
- Otherwise, initializes with random/packed configuration.
"""
function initialize_state(
    params::Parameters,
    pathname::String;
    from_file::String="",
    dimension::Int=3,
    random_init=false,
    cutoff=1.5,
    rng::Random.AbstractRNG=Random.Xoshiro(),
    positions=nothing,
    diameters=nothing,
    box_type::Union{String,Symbol}="cubic",
    box_size=nothing,
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
        positions=positions,
        diameters=diameters,
        box_type=box_type,
        box_size=box_size,
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
    @info boxl
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