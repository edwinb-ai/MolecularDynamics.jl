function reset_output!(output::EnergyAndForces)
    output.energy = 0.0
    output.virial = 0.0

    @inbounds for f in output.forces
        fill!(f, zero(Float64))
    end

    return nothing
end

"""
    _energy_and_forces!(
        x, y, neighborlist, output, potential, diameters, buffers
    )

Computation using a neighbor list produced by CellListMap.neighborlist(x, y, cutoff).
- x, y: position arrays (vectors of positions)
- neighborlist: vector of (i, j, dist) tuples from CellListMap
- output: EnergyAndForces struct with .energy, .virial, .forces (pre-allocated)
- potential: user-defined potential with `evaluate(potential, dist, σ1, σ2)`
- diameters: vector of particle diameters
- buffers: ThreadLocalBuffers, to be reused across time steps
"""
function energy_and_forces!(
    x, neighborlist, output, potential, diameters, unitcell, unitcell_inv
)
    @inbounds @simd for k in eachindex(neighborlist)
        (i, j, _) = neighborlist[k]
        xi = x[i]
        xj = x[j]
        σ1 = diameters[i]
        σ2 = diameters[j]
        rvec = xi - xj
        minimum_image!(rvec, unitcell, unitcell_inv)
        dist = norm(rvec)
        (uij, fij) = evaluate(potential, dist, σ1, σ2)
        fij_vec = @. fij * rvec / dist

        output.energy += uij
        output.virial += dot(fij_vec, rvec)
        output.forces[i] .+= fij_vec
        output.forces[j] .-= fij_vec
    end

    return nothing
end
