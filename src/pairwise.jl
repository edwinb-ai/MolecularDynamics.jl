"Custom copy, reset and reducer functions"
function copy_output(x::EnergyAndForces)
    return EnergyAndForces(copy(x.energy), copy(x.virial), copy(x.forces))
end

function reset_output!(output::EnergyAndForces)
    output.energy = 0.0
    output.virial = 0.0

    for i in eachindex(output.forces)
        output.forces[i] = StaticArrays.SVector(0.0, 0.0, 0.0)
    end

    return output
end

function reducer(x::EnergyAndForces, y::EnergyAndForces)
    e_tot = x.energy + y.energy
    vir_tot = x.virial + y.virial
    x.forces .+= y.forces

    return EnergyAndForces(e_tot, vir_tot, x.forces)
end

"Function that updates energy and forces for each pair"
function energy_and_forces!(
    x, y, i, j, d2, output::EnergyAndForces, pot::T
) where {T<:Potential}
    d = sqrt(d2)
    r = x - y
    (uij, fij) = pot.potf(d)
    sumies = @. fij * r / d
    output.virial += dot(sumies, r)
    output.energy += uij
    output.forces[i] = @. output.forces[i] + sumies
    output.forces[j] = @. output.forces[j] - sumies

    return output
end
