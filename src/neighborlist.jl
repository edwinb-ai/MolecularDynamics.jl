"""
    NeighborChunk

Rows of the half neighbor list for a contiguous range of bins, in compressed sparse row
form. Chunks are the units of parallel work.
"""
struct NeighborChunk
    # Particle index of each row
    atoms::Vector{Int32}
    # Row `r` spans `neighbors[start[r]:(start[r + 1] - 1)]`
    start::Vector{Int32}
    neighbors::Vector{Int32}
    # Index into the lattice shift table, one per neighbor
    codes::Vector{Int32}
    # Scratch space: (first, last, shift code) of each range of binned particles searched
    segments::Vector{NTuple{3,Int32}}
end

NeighborChunk() = NeighborChunk(Int32[], Int32[1], Int32[], Int32[], NTuple{3,Int32}[])

"""
    NeighborList(positions, unitcell, cutoff; skin=0.3, nchunks)

Verlet half neighbor list holding every periodic image within `cutoff + skin`, built
from a binned cell list. Each pair stores the lattice translation of its image, so the
force loop needs no minimum-image arithmetic and any (triclinic) box and box size works.

The list stays valid until some particle has moved more than `skin / 2` since the last
[`build!`](@ref); see [`update!`](@ref). It is split into `nchunks` parts that `ntasks`
parallel tasks pick up dynamically, so faster cores take more of the work. Each task
accumulates forces into its own buffer.
"""
mutable struct NeighborList{D,T<:AbstractFloat,M}
    cutoff::T
    skin::T
    unitcell::M
    unitcell_inv::M
    # Number of bins along each lattice direction
    nbins::NTuple{D,Int}
    # Rows of bins searched from each bin: an offset along directions 2:D (first entry
    # unused) and a range of offsets along direction 1. Bins along direction 1 are
    # contiguous in memory. The first row is the bin's own, searched forward only.
    rows::Vector{Tuple{NTuple{D,Int},UnitRange{Int}}}
    shift_range::NTuple{D,Int}
    # Lattice translations, indexed by shift code
    shifts::Vector{SVector{D,T}}
    # Particles sorted by bin, in compressed sparse row form
    bin_start::Vector{Int32}
    bin_atoms::Vector{Int32}
    atom_bin::Vector{Int32}
    # Wrapped positions in bin order
    binned::Vector{SVector{D,T}}
    chunk_bins::Vector{UnitRange{Int}}
    chunks::Vector{NeighborChunk}
    # Per-task force buffers and (energy, virial) of the last force evaluation
    buffers::Vector{Vector{SVector{D,T}}}
    partial::Vector{NTuple{2,T}}
    # Next chunk to be picked up by a task
    next_chunk::Threads.Atomic{Int}
    # Positions at the last build
    reference::Vector{SVector{D,T}}
    nbuilds::Int
end

# Several chunks per thread for load balancing, but at least ~64 particles per chunk
default_nchunks(n_particles) = clamp(n_particles ÷ 64, 1, 4 * Threads.nthreads())

function NeighborList(
    positions::AbstractVector{SVector{D,T}},
    unitcell,
    cutoff;
    skin=0.3,
    nchunks::Int=default_nchunks(length(positions)),
    ntasks::Int=min(Threads.nthreads(), nchunks),
) where {D,T}
    cutoff > 0 || throw(ArgumentError("cutoff must be positive, got $cutoff"))
    skin >= 0 || throw(ArgumentError("skin must be non-negative, got $skin"))
    nchunks >= 1 || throw(ArgumentError("nchunks must be at least 1, got $nchunks"))
    ntasks >= 1 || throw(ArgumentError("ntasks must be at least 1, got $ntasks"))

    uc = SMatrix{D,D,T}(unitcell)
    uc_inv = inv(uc)
    r_list = cutoff + skin
    # Bins are at least r_list / 2 wide, measured perpendicular to the box faces
    heights = ntuple(k -> 1 / norm(uc_inv[k, :]), D)
    nbins = ntuple(k -> max(1, floor(Int, 2 * heights[k] / r_list)), D)
    widths = heights ./ nbins
    reach = ntuple(k -> ceil(Int, r_list / widths[k]), D)

    # Lower bound of the distance between two bins, given their offset
    orthorhombic = isdiag(uc)
    gap(o, k) = max(abs(o) - 1, 0) * widths[k]
    min_distance(gaps) = orthorhombic ? sqrt(sum(abs2, gaps)) : maximum(gaps)

    # Keep one of each pair of opposite offsets, dropping bins that are surely too far
    rows = Tuple{NTuple{D,Int},UnitRange{Int}}[]
    for offset in CartesianIndices(ntuple(k -> k == 1 ? (0:0) : (-reach[k]):reach[k], D))
        o = Tuple(offset)
        own_row = all(iszero, o)
        (own_row || is_forward(o)) || continue
        m = -1
        for o1 in 0:reach[1]
            min_distance(ntuple(k -> gap(k == 1 ? o1 : o[k], k), D)) < r_list || break
            m = o1
        end
        m < 0 && continue
        push!(rows, (o, own_row ? (0:m) : ((-m):m)))
    end
    sort!(rows; by=row -> !all(iszero, first(row)))

    shift_range = ntuple(k -> cld(reach[k], nbins[k]), D)
    shifts = vec([
        uc * SVector{D,T}(Tuple(t)) for
        t in CartesianIndices(ntuple(k -> (-shift_range[k]):shift_range[k], D))
    ])

    n_particles = length(positions)
    return NeighborList{D,T,typeof(uc)}(
        T(cutoff),
        T(skin),
        uc,
        uc_inv,
        nbins,
        rows,
        shift_range,
        shifts,
        zeros(Int32, prod(nbins) + 1),
        zeros(Int32, n_particles),
        zeros(Int32, n_particles),
        copy(positions),
        fill(1:0, nchunks),
        [NeighborChunk() for _ in 1:nchunks],
        [zeros(SVector{D,T}, n_particles) for _ in 1:ntasks],
        fill((zero(T), zero(T)), ntasks),
        Threads.Atomic{Int}(1),
        copy(positions),
        0,
    )
end

# Last non-zero component is positive
@inline function is_forward(o::NTuple{D,Int}) where {D}
    for k in D:-1:1
        o[k] != 0 && return o[k] > 0
    end
    return false
end

# 1-based linear index of the 0-based Cartesian coordinates `c` in an array of size `dims`
@inline function linear_index(c::NTuple{D,Int}, dims::NTuple{D,Int}) where {D}
    index = 1
    stride = 1
    for k in 1:D
        index += c[k] * stride
        stride *= dims[k]
    end
    return index
end

# 0-based Cartesian coordinates of the linear bin index `b`
@inline function bin_coordinates(b::Int, dims::NTuple{D,Int}) where {D}
    return Tuple(CartesianIndices(dims)[b]) .- 1
end

# Index in the shift table of the lattice translation `t` (in box vectors)
@inline function shift_code(t::NTuple{D,Int}, range::NTuple{D,Int}) where {D}
    return linear_index(t .+ range, 2 .* range .+ 1)
end

"""
    needs_rebuild(nl, positions) -> Bool

Whether any particle has moved more than half the skin since the last build.
"""
function needs_rebuild(nl::NeighborList, positions)
    threshold = (nl.skin / 2)^2
    @inbounds for i in eachindex(positions, nl.reference)
        sum(abs2, positions[i] - nl.reference[i]) > threshold && return true
    end
    return false
end

"""
    update!(nl, positions, images) -> Bool

Rebuild the neighbor list if needed. Returns `true` if it was rebuilt.
"""
function update!(nl::NeighborList, positions, images)
    if needs_rebuild(nl, positions)
        build!(nl, positions, images)
        return true
    end
    return false
end

"""
    build!(nl, positions, images)

Wrap `positions` into the box (updating `images`) and rebuild the neighbor list.
"""
function build!(nl::NeighborList{D}, positions, images) where {D}
    wrap_positions!(positions, images, nl.unitcell, nl.unitcell_inv)
    n_particles = length(positions)
    resize!(nl.reference, n_particles)
    copyto!(nl.reference, positions)

    # Counting sort of the particles into bins
    nbins = nl.nbins
    bin_start = nl.bin_start
    atom_bin = resize!(nl.atom_bin, n_particles)
    fill!(bin_start, 0)
    @inbounds for i in eachindex(positions)
        s = nl.unitcell_inv * positions[i]
        b = linear_index(
            ntuple(k -> clamp(floor(Int, s[k] * nbins[k]), 0, nbins[k] - 1), D), nbins
        )
        atom_bin[i] = b
        bin_start[b + 1] += 1
    end
    bin_start[1] = 1
    for b in 1:(length(bin_start) - 1)
        bin_start[b + 1] += bin_start[b]
    end
    bin_atoms = resize!(nl.bin_atoms, n_particles)
    @inbounds for i in eachindex(positions)
        b = atom_bin[i]
        bin_atoms[bin_start[b]] = i
        bin_start[b] += 1
    end
    # Restore the start offsets shifted by the fill above
    @inbounds for b in (length(bin_start) - 1):-1:2
        bin_start[b] = bin_start[b - 1]
    end
    bin_start[1] = 1
    binned = resize!(nl.binned, n_particles)
    @inbounds for p in eachindex(binned, bin_atoms)
        binned[p] = positions[bin_atoms[p]]
    end

    # Contiguous ranges of bins with about the same number of particles per chunk
    nchunks = length(nl.chunks)
    lo = 1
    c = 1
    for b in 1:(length(bin_start) - 1)
        if c < nchunks && bin_start[b + 1] - 1 >= c * n_particles / nchunks
            nl.chunk_bins[c] = lo:b
            lo = b + 1
            c += 1
        end
    end
    for k in c:nchunks
        nl.chunk_bins[k] = k == c ? (lo:(length(bin_start) - 1)) : (1:0)
    end

    foreach_chunk(nl) do _, c
        return build_chunk!(nl.chunks[c], nl, nl.chunk_bins[c])
    end
    for buffer in nl.buffers
        resize!(buffer, n_particles)
    end
    nl.nbuilds += 1

    return nl
end

"""
    foreach_chunk(f, nl)

Call `f(task, chunk_index)` for every chunk, with `length(nl.buffers)` tasks pulling
chunks from a shared counter.
"""
function foreach_chunk(f::F, nl::NeighborList) where {F}
    nchunks = length(nl.chunks)
    ntasks = length(nl.buffers)
    if ntasks == 1
        for c in 1:nchunks
            f(1, c)
        end
        return nothing
    end
    nl.next_chunk[] = 1
    @sync for task in 1:ntasks
        Threads.@spawn while true
            c = Threads.atomic_add!(nl.next_chunk, 1)
            c > nchunks && break
            f(task, c)
        end
    end
    return nothing
end

"""
    build_chunk!(chunk, nl, bins)

Fill the rows of `chunk` with the neighbors of every particle in `bins`. Arrays are sized
up front and filled by index; `push!` in the inner loops made the build twice as slow.
"""
function build_chunk!(chunk::NeighborChunk, nl::NeighborList{D}, bins) where {D}
    (; atoms, start, neighbors, codes, segments) = chunk
    (; nbins, rows, shift_range, shifts, bin_start, bin_atoms, binned) = nl
    r2_list = (nl.cutoff + nl.skin)^2
    n1 = nbins[1]

    natoms = isempty(bins) ? 0 : Int(bin_start[last(bins) + 1] - bin_start[first(bins)])
    resize!(atoms, natoms)
    resize!(start, natoms + 1)
    start[1] = 1
    # Each row is split at most once per period along direction 1
    max_segments = sum(row -> cld(length(last(row)), n1) + 1, rows)
    length(segments) < max_segments && resize!(segments, max_segments)
    row = 0
    n = 0

    @inbounds for b in bins
        first_in_bin = Int(bin_start[b])
        last_in_bin = Int(bin_start[b + 1]) - 1
        first_in_bin > last_in_bin && continue

        # Ranges of binned particles to search from this bin, split where the lattice
        # shift along direction 1 changes
        nsegments = 0
        candidates = 0
        coordinates = bin_coordinates(b, nbins)
        for (o, xrange) in rows
            unrolled = coordinates .+ o
            row_bins = mod.(unrolled, nbins)
            row_shifts = fld.(unrolled, nbins)
            u = coordinates[1] + first(xrange)
            u_last = coordinates[1] + last(xrange)
            while u <= u_last
                t1 = fld(u, n1)
                segment_last = min(u_last, (t1 + 1) * n1 - 1)
                bin_first = linear_index(Base.setindex(row_bins, u - t1 * n1, 1), nbins)
                bin_last = bin_first + (segment_last - u)
                code = shift_code(Base.setindex(row_shifts, t1, 1), shift_range)
                q_first = bin_start[bin_first]
                q_last = bin_start[bin_last + 1] - 1
                nsegments += 1
                segments[nsegments] = (q_first, q_last, Int32(code))
                candidates += q_last - q_first + 1
                u = segment_last + 1
            end
        end

        # Room for every candidate of every particle in this bin
        needed = n + (last_in_bin - first_in_bin + 1) * candidates
        if length(neighbors) < needed
            resize!(neighbors, max(needed, 2 * length(neighbors)))
            resize!(codes, length(neighbors))
        end

        for p in first_in_bin:last_in_bin
            xi = binned[p]
            row += 1
            atoms[row] = bin_atoms[p]
            for s in 1:nsegments
                (q_first, q_last, code) = segments[s]
                # The first segment starts at this bin: take each pair once
                q_start = s == 1 ? p + 1 : Int(q_first)
                xi_shifted = xi - shifts[code]
                for q in q_start:q_last
                    r = xi_shifted - binned[q]
                    if dot(r, r) < r2_list
                        n += 1
                        neighbors[n] = bin_atoms[q]
                        codes[n] = code
                    end
                end
            end
            start[row + 1] = n + 1
        end
    end
    resize!(neighbors, n)
    resize!(codes, n)

    return chunk
end

"""
    npairs(nl) -> Int

Total number of pairs stored in the neighbor list.
"""
npairs(nl::NeighborList) = sum(c -> length(c.neighbors), nl.chunks)
