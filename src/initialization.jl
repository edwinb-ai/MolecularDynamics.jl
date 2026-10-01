"""
    to_unitcell(box, dimension) -> SMatrix

Convert box to a `dimension × dimension` static matrix whose columns are the box vectors.
Accepts a scalar (creates cubic), vector (diagonal), or matrix (full).
"""
function to_unitcell(box, dimension)
    if isa(box, Number)
        return SMatrix{dimension,dimension,Float64}(box * I)
    elseif isa(box, AbstractVector)
        return SMatrix{dimension,dimension,Float64}(Diagonal(box))
    elseif isa(box, AbstractMatrix)
        # Extract upper-left dimension x dimension in case it's bigger
        return SMatrix{dimension,dimension,Float64}(box[1:dimension, 1:dimension])
    else
        error("Cannot interpret box/unitcell of type $(typeof(box))")
    end
end

"""
    initialize_random(unitcell, npart, rng, dimension; tol=1.0)

Random positions inside an orthorhombic box, packed with Packmol to remove overlaps.
"""
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

"""
    initialize_velocities(ktemp, rng, n_particles, dimension) -> Vector{SVector}

Gaussian velocities with zero total momentum, scaled to exactly temperature `ktemp`.
"""
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
    velocities = [SVector{dimension,Float64}(view(V, :, i)) for i in 1:n_particles]

    return velocities
end

"""
    initialize_simulation(params, rng, dimension; kwargs...) -> (positions, unitcell, diameters)

Positions, box and diameters from user input, a file or a random packing. See
[`initialize_state`](@ref) for the keyword arguments.
"""
function initialize_simulation(
    params::Parameters,
    rng,
    dimension;
    from_file::String="",
    random_init::Bool=false,
    unitcell=nothing,
    positions=nothing,
    diameters=nothing,
)
    n_particles = params.n_particles

    # If user provides positions and diameters, use them directly
    if positions !== nothing && diameters !== nothing
        @info "Initializing system from user-provided positions and diameters."
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
        unitcell = to_unitcell(unitcell, dimension)
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

    positions = [SVector{dimension,Float64}(x) for x in positions]
    diameters = Vector{Float64}(diameters)

    return positions, unitcell, diameters
end

"""
    initialize_state(params, pathname; kwargs...) -> SimulationState

Create the simulation state and write the initial configuration to `pathname/init.xyz`.
Velocities are left empty; set them with `state.velocities = initialize_velocities(...)`.

# Keyword arguments
- `dimension=3`: 2 or 3.
- `cutoff=1.5`: interaction cutoff used by the neighbor list.
- `skin=0.3`: neighbor list skin; larger values rebuild less often but store more pairs.
- `from_file=""`, `random_init=false`, `unitcell`, `positions`, `diameters`: how the
  configuration is created, see [`initialize_simulation`](@ref).
- `rng=Random.Xoshiro()`: random number generator used by the whole simulation.
"""
function initialize_state(
    params::Parameters,
    pathname::String;
    from_file::String="",
    dimension::Int=3,
    random_init=false,
    cutoff=1.5,
    skin=0.3,
    rng::AbstractRNG=Random.Xoshiro(),
    unitcell=nothing,
    positions=nothing,
    diameters=nothing,
)
    nf = dimension * (params.n_particles - 1.0)
    (positions, unitcell, diameters) = initialize_simulation(
        params,
        rng,
        dimension;
        from_file=from_file,
        random_init=random_init,
        unitcell=unitcell,
        positions=positions,
        diameters=diameters,
    )

    n_particles = length(positions)
    images = zeros(SVector{dimension,Int32}, n_particles)
    neighbors = NeighborList(positions, unitcell, cutoff; skin=skin)
    build!(neighbors, positions, images)

    state = SimulationState(
        positions,
        SVector{dimension,Float64}[],
        zeros(SVector{dimension,Float64}, n_particles),
        images,
        diameters,
        unitcell,
        rng,
        neighbors,
        dimension,
        nf,
        0.0,
        0.0,
    )

    # Write initial configuration
    write_to_file(
        joinpath(pathname, "init.xyz"),
        0,
        unitcell,
        n_particles,
        positions,
        diameters,
        dimension;
        mode="w",
    )

    return state
end
