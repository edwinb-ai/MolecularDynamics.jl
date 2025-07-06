"""
    to_unitcell(box, dimension) -> SMatrix

Convert box to a StaticArray SMatrix of size (dimension, dimension).
Accepts a scalar (creates cubic), vector (diagonal), or matrix (full).
"""
function to_unitcell(box, dimension)
    if isa(box, Number)
        return box * SMatrix{dimension,dimension,Float64}(I)
    elseif isa(box, AbstractVector)
        return SMatrix{dimension,dimension,Float64}(Diagonal(box))
    elseif isa(box, AbstractMatrix)
        # Extract upper-left dimension x dimension in case it's bigger
        return SMatrix{dimension,dimension,Float64}(box[1:dimension, 1:dimension])
    else
        error("Cannot interpret box/unitcell of type $(typeof(box))")
    end
end

function initialize_random(unitcell, npart, rng, dimension; tol=1.0)
    # Assume unitcell is a matrix, generate random positions within the box
    mins = zeros(dimension)
    maxs = diag(unitcell)
    coordinates = [
        SVector{dimension,Float64}(rand(rng, Float64, dimension) .* (maxs .- mins) .+ mins)
        for _ in 1:npart
    ]
    pack_monoatomic!(coordinates, maxs, tol; parallel=true, iprint=100)
    return coordinates
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

function initialize_simulation(
    params::Parameters,
    rng,
    dimension;
    cutoff::Float64=1.5,
    from_file::String="",
    random_init::Bool=false,
    unitcell=nothing,
    positions=nothing,
    diameters=nothing,
)
    system = nothing
    n_particles = params.n_particles

    # If user provides positions and diameters, use them directly
    if positions !== nothing && diameters !== nothing
        @info "Initializing system from user-provided positions and diameters."
        n_particles = length(positions)
        if unitcell === nothing
            # Try to infer a bounding box from the positions if not given
            # (This is a naive approach, consider improving)
            mins = mapreduce(x -> minimum(x), min, positions)
            maxs = mapreduce(x -> maximum(x), max, positions)
            box_vec = maxs .- mins
            unitcell = to_unitcell(box_vec, dimension)
        else
            unitcell = to_unitcell(unitcell, dimension)
        end
    elseif isfile(from_file) || !random_init
        @info "Reading from file..."
        (unitcell, positions, diameters) = read_file(from_file; dimension=dimension)
        n_particles = length(positions)
    elseif unitcell !== nothing
        # User provided a box/unitcell (matrix, vector, or scalar)
        unitcell = to_unitcell(unitcell, dimension)
        positions = initialize_random(unitcell, n_particles, rng, dimension)
        diameters = ones(n_particles)
    else
        # Default cubic/square box
        @info "Initializing random positions in a box of dimension $dimension ."
        boxl = (n_particles / params.ρ)^(1.0 / dimension)
        unitcell = to_unitcell(boxl, dimension)
        positions = initialize_random(unitcell, n_particles, rng, dimension)
        diameters = ones(n_particles)
    end

    system = CellListMap.ParticleSystem(;
        xpositions=positions,
        unitcell=unitcell,
        cutoff=cutoff,
        output=EnergyAndForces(0.0, 0.0, similar(positions)),
        output_name=:energy_and_forces,
        parallel=true,
    )

    return system, unitcell, diameters
end

function initialize_state(
    params::Parameters,
    pathname::String;
    from_file::String="",
    dimension::Int=3,
    random_init=false,
    cutoff=1.5,
    rng::Random.AbstractRNG=Random.Xoshiro(),
    unitcell=nothing,
    positions=nothing,
    diameters=nothing,
)
    nf = dimension * (params.n_particles - 1.0)
    (system, unitcell, diameters) = initialize_simulation(
        params,
        rng,
        dimension;
        cutoff=cutoff,
        from_file=from_file,
        random_init=random_init,
        unitcell=unitcell,
        positions=positions,
        diameters=diameters,
    )

    for i in eachindex(system.energy_and_forces.forces)
        system.energy_and_forces.forces[i] = zeros(SVector{dimension})
    end
    reset_output!(system.energy_and_forces)

    images = [zeros(MVector{dimension,Int32}) for _ in eachindex(system.xpositions)]

    state = SimulationState(
        system,
        diameters,
        rng,
        unitcell,
        Vector{SVector{dimension,Float64}}(),
        images,
        dimension,
        nf,
    )

    # Write initial configuration
    write_to_file(
        joinpath(pathname, "init.xyz"),
        0,
        unitcell,
        length(system.positions),
        system.positions,
        diameters,
        dimension;
        mode="w",
    )

    return state
end
