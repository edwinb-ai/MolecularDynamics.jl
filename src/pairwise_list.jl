function reset_output!(output::EnergyAndForces)
    output.energy = 0.0
    output.virial = 0.0
    fill!(output.forces, zeros(eltype(output.forces)))

    return nothing
end

"""
    ThreadLocalBuffers

Caches thread-local accumulators to avoid repeated allocations during parallel force/energy computation.
- forces_tls: Vector of force arrays, one per thread (nthreads × n_particles).
- energy_tls: Vector of energies, one per thread.
- virial_tls: Vector of virials, one per thread.
"""
mutable struct ThreadLocalBuffers{T}
    forces_tls::Vector{Vector{T}}
    energy_tls::Vector{Float64}
    virial_tls::Vector{Float64}
end

"""
    init_thread_local_buffers(n_particles::Int, T::Type=Float64)

Initialize thread-local buffers for parallel force/energy computation.
"""
function init_thread_local_buffers(n_particles::Int, dimension::Int, T::Type=Float64)
    nthreads = Threads.nthreads()
    forces_tls = [[zero(MVector{dimension,T}) for _ in 1:n_particles] for _ in 1:nthreads]
    energy_tls = zeros(Float64, nthreads)
    virial_tls = zeros(Float64, nthreads)
    return ThreadLocalBuffers(forces_tls, energy_tls, virial_tls)
end

"""
    reset_thread_local_buffers!(buffers::ThreadLocalBuffers, n_particles::Int)

Reset all accumulators to zero before each parallel computation.
"""
function reset_thread_local_buffers!(buffers::ThreadLocalBuffers, n_particles::Int)
    @inbounds for fvec in buffers.forces_tls
        @simd for fi in fvec
            fill!(fi, 0.0)
        end
    end
    fill!(buffers.energy_tls, 0.0)
    fill!(buffers.virial_tls, 0.0)

    return nothing
end

"""
    parallel_energy_and_forces!(
        x, y, neighborlist, output, potential, diameters, buffers
    )

Parallel computation using a neighbor list produced by CellListMap.neighborlist(x, y, cutoff).
- x, y: position arrays (vectors of positions)
- neighborlist: vector of (i, j, dist) tuples from CellListMap
- output: EnergyAndForces struct with .energy, .virial, .forces (pre-allocated)
- potential: user-defined potential with `evaluate(potential, dist, σ1, σ2)`
- diameters: vector of particle diameters
- buffers: ThreadLocalBuffers, to be reused across time steps
"""
function energy_and_forces!(
    x, y, neighborlist, output, potential, diameters, buffers::ThreadLocalBuffers
)
    n = length(x)
    nthreads = Threads.nthreads()

    reset_thread_local_buffers!(buffers, n)

    Threads.@threads for k in eachindex(neighborlist)
        tid = threadid()
        (i, j, dist) = neighborlist[k]
        xi = x[i]
        xj = y[j]
        σ1 = diameters[i]
        σ2 = diameters[j]
        rvec = xj - xi
        (uij, fij) = evaluate(potential, dist, σ1, σ2)
        fij_vec = @. fij * rvec / dist

        buffers.energy_tls[tid] += uij
        buffers.virial_tls[tid] += dot(fij_vec, rvec)
        buffers.forces_tls[tid][i] .+= fij_vec
        buffers.forces_tls[tid][j] .-= fij_vec
    end

    # Reduce thread-local results into output
    vector_size = length(output.forces[1])
    for tid in 1:nthreads
        output.energy += buffers.energy_tls[tid]
        output.virial += buffers.virial_tls[tid]
        @inbounds for i in eachindex(output.forces)
            output.forces[i] = SVector{vector_size}(buffers.forces_tls[tid][i]) .+ output.forces[i]
        end
    end

    return nothing
end
