"""
    compute_forces!(forces, positions, diameters, potential, nl) -> (energy, virial)

Evaluate pair forces over the neighbor list `nl`, overwriting `forces`. Returns the total
potential energy and the virial `Σ r_ij ⋅ f_ij`. Parallel tasks accumulate into their
own force buffers, which are summed at the end.
"""
function compute_forces!(forces, positions, diameters, potential, nl::NeighborList)
    rc2 = nl.cutoff^2
    chunks = nl.chunks
    if length(nl.buffers) == 1
        fill!(forces, zero(eltype(forces)))
        energy = zero(eltype(diameters))
        virial = zero(eltype(diameters))
        for chunk in chunks
            (e, w) = chunk_forces!(
                forces, positions, diameters, potential, chunk, nl.shifts, rc2
            )
            energy += e
            virial += w
        end
        return energy, virial
    end

    fill!(nl.partial, (zero(eltype(diameters)), zero(eltype(diameters))))
    @threads for buffer in nl.buffers
        fill!(buffer, zero(eltype(buffer)))
    end
    foreach_chunk(nl) do task, c
        (e, w) = chunk_forces!(
            nl.buffers[task], positions, diameters, potential, chunks[c], nl.shifts, rc2
        )
        return nl.partial[task] = nl.partial[task] .+ (e, w)
    end
    reduce_forces!(forces, nl.buffers)

    energy = sum(first, nl.partial)
    virial = sum(last, nl.partial)
    return energy, virial
end

"""
    compute_forces!(state, potential)

Evaluate forces for `state`, storing the potential energy and virial in it.
"""
function compute_forces!(state::SimulationState, potential)
    (state.energy, state.virial) = compute_forces!(
        state.forces, state.positions, state.diameters, potential, state.neighbors
    )
    return state
end

"""
    chunk_forces!(forces, positions, diameters, potential, chunk, shifts, rc2) -> (energy, virial)

Accumulate the pair forces of the rows in `chunk` into `forces`, for pairs closer than
`sqrt(rc2)`.
"""
function chunk_forces!(forces, positions, diameters, potential, chunk, shifts, rc2)
    (; atoms, start, neighbors, codes) = chunk
    energy = zero(eltype(diameters))
    virial = zero(eltype(diameters))
    @inbounds for row in eachindex(atoms)
        i = atoms[row]
        xi = positions[i]
        di = diameters[i]
        fi = zero(eltype(forces))
        for k in start[row]:(start[row + 1] - 1)
            j = neighbors[k]
            r = xi - positions[j] - shifts[codes[k]]
            r2 = dot(r, r)
            if r2 < rc2
                (uij, fij_over_r) = evaluate_r2(potential, r2, di, diameters[j])
                fij = fij_over_r * r
                fi += fij
                forces[j] -= fij
                energy += uij
                virial += fij_over_r * r2
            end
        end
        forces[i] += fi
    end
    return energy, virial
end

"""
    reduce_forces!(forces, buffers)

Sum the per-task force `buffers` into `forces`, in parallel over particles.
"""
function reduce_forces!(forces, buffers)
    n = length(forces)
    nparts = length(buffers)
    @threads for c in 1:nparts
        @inbounds for i in (div((c - 1) * n, nparts) + 1):div(c * n, nparts)
            f = buffers[1][i]
            for k in 2:nparts
                f += buffers[k][i]
            end
            forces[i] = f
        end
    end
    return forces
end
