"""
    radial_derivatives(potential, r, sigma1, sigma2) -> (dU/dr, d²U/dr²)

Exact first and second derivatives of the pair energy at distance `r`, computed with
ForwardDiff through `evaluate`.
"""
function radial_derivatives(potential, r, sigma1, sigma2)
    energy(s) = first(evaluate(potential, s, sigma1, sigma2))
    slope(s) = ForwardDiff.derivative(energy, s)
    return slope(r), ForwardDiff.derivative(slope, r)
end

"""
    hessian(state, potential; check_forces=true) -> SparseMatrixCSC
    hessian(state, params::Parameters; check_forces=true)

Hessian of the potential energy, `H[d(i-1)+α, d(j-1)+β] = ∂²U / ∂x_{iα} ∂x_{jβ}`, as a
sparse symmetric `dN × dN` matrix (`d` the dimension). The first and second radial
derivatives of every pair within the cutoff come from ForwardDiff, so any potential works
as long as its `evaluate` method accepts `r::Real`. Periodic images, triclinic boxes and
polydispersity are handled as in the force calculation; `state` is not modified.

With `check_forces=true` it warns when the force returned by `evaluate` is not `-dU/dr`.
"""
function hessian(
    state::SimulationState{D,T}, potential::Potential; check_forces::Bool=true
) where {D,T}
    # A list without skin holds exactly the pairs within the cutoff
    x = copy(state.positions)
    images = zeros(SVector{D,Int32}, length(x))
    nl = NeighborList(
        x, state.unitcell, state.neighbors.cutoff; skin=0.0, nchunks=1, ntasks=1
    )
    build!(nl, x, images)
    diameters = state.diameters
    rc2 = nl.cutoff^2
    N = length(x)

    diagonal = zeros(SMatrix{D,D,T,D * D}, N)
    rows = Int[]
    columns = Int[]
    values = T[]
    sizehint!.((rows, columns, values), D^2 * (N + 2 * npairs(nl)))
    (worst_mismatch, worst_r) = (zero(T), zero(T))

    for chunk in nl.chunks
        (; atoms, start, neighbors, codes) = chunk
        for row in eachindex(atoms)
            i = atoms[row]
            for k in start[row]:(start[row + 1] - 1)
                j = neighbors[k]
                r = x[i] - x[j] - nl.shifts[codes[k]]
                r2 = dot(r, r)
                r2 < rc2 || continue
                d = sqrt(r2)
                (du, d2u) = radial_derivatives(potential, d, diameters[i], diameters[j])

                if check_forces
                    f = last(evaluate(potential, d, diameters[i], diameters[j]))
                    mismatch = abs(f + du) / max(one(T), abs(du))
                    if mismatch > worst_mismatch
                        (worst_mismatch, worst_r) = (mismatch, d)
                    end
                end

                # Second derivative of u(|r|) with respect to r
                nn = (r / d) * (r / d)'
                K = d2u * nn + (du / d) * (one(nn) - nn)
                diagonal[i] += K
                diagonal[j] += K
                push_block!(rows, columns, values, i, j, -K)
                push_block!(rows, columns, values, j, i, -K)
            end
        end
    end
    for i in 1:N
        push_block!(rows, columns, values, i, i, diagonal[i])
    end

    if worst_mismatch > 1e-6
        @warn "The force returned by `evaluate` differs from -dU/dr (relative difference " *
            "$(worst_mismatch) at r = $(worst_r)); the Hessian follows the energy."
    end

    # Repeated entries, e.g. several images of the same pair, are summed
    return sparse(rows, columns, values, D * N, D * N)
end

function hessian(state::SimulationState, params::Parameters; kwargs...)
    return hessian(state, params.potential; kwargs...)
end

function push_block!(rows, columns, values, i, j, block::SMatrix{D,D}) where {D}
    for β in 1:D, α in 1:D
        push!(rows, D * (i - 1) + α)
        push!(columns, D * (j - 1) + β)
        push!(values, block[α, β])
    end
    return nothing
end

"""
    normal_modes(H) -> (frequencies, modes)

All vibrational modes of the Hessian `H` for unit masses, by dense diagonalization.
Frequencies are `sqrt(λ)` in increasing order, negative where the eigenvalue `λ` is
negative (unstable directions), and `modes[:, k]` is the normalized eigenvector of mode
`k`. The cost grows as `(dN)³`, which is practical up to `dN` of about 10⁴.
"""
function normal_modes(H::AbstractMatrix)
    (λ, modes) = eigen(Symmetric(Matrix(H)))
    frequencies = @. sign(λ) * sqrt(abs(λ))
    return frequencies, modes
end

"""
    lowest_modes(H, k; shift=-1e-4, tol=1e-10) -> (frequencies, modes)

The `k` lowest vibrational modes of the sparse Hessian `H` (unit masses), in the same form
as [`normal_modes`](@ref). Uses shift-invert Lanczos: the largest eigenvalues of
`(H - shift I)⁻¹`, with a sparse Cholesky factorization. `shift` must lie below the lowest
eigenvalue; it is negative because translations have zero frequency. Far cheaper than
`normal_modes` for large systems, e.g. a few seconds for 50 modes of 65536 particles.
"""
function lowest_modes(H::SparseMatrixCSC, k::Integer; shift=-1e-4, tol=1e-10)
    n = size(H, 1)
    1 <= k <= n || throw(ArgumentError("k must be between 1 and $n, got $k"))
    F = cholesky(Symmetric(H - shift * I); check=false)
    if !issuccess(F)
        throw(
            ArgumentError(
                "H - shift * I is not positive definite: the configuration has eigenvalues " *
                "below shift = $shift. Minimize it further or use a more negative shift.",
            ),
        )
    end
    start = rand(Xoshiro(0), n)
    (μ, vectors, info) = eigsolve(
        x -> F \ x, start, k, :LR; ishermitian=true, krylovdim=max(2k, 30), tol=tol
    )
    if info.converged < k
        @warn "Only $(info.converged) of $k modes converged."
    end
    λ = shift .+ 1 ./ μ[1:k]
    order = sortperm(λ)
    frequencies = @. sign(λ[order]) * sqrt(abs(λ[order]))
    return frequencies, reduce(hcat, vectors[order])
end

"""
    participation_ratio(mode, dimension) -> Float64
    participation_ratio(modes::AbstractMatrix, dimension) -> Vector{Float64}

Participation ratio `P = (Σᵢ |eᵢ|²)² / (N Σᵢ |eᵢ|⁴)` of a mode, `eᵢ` being the displacement
of particle `i`. It is about 1 for extended modes and of order `1/N` for modes localized on
a few particles, such as the quasi-localized modes of glasses.
"""
function participation_ratio(mode::AbstractVector, dimension::Integer)
    N = length(mode) ÷ dimension
    weights = [
        sum(abs2, view(mode, (dimension * (i - 1) + 1):(dimension * i))) for i in 1:N
    ]
    return sum(weights)^2 / (N * sum(abs2, weights))
end

function participation_ratio(modes::AbstractMatrix, dimension::Integer)
    return [participation_ratio(view(modes, :, k), dimension) for k in axes(modes, 2)]
end
