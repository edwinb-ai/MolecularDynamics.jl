function reset_output!(output::EnergyAndForces)
    output.energy = 0.0
    output.virial = 0.0

    @inbounds for f in output.forces
        fill!(f, zero(Float64))
    end

    return output
end

"Function that updates energy and forces for each pair"
function energy_and_forces!(
    x, y, i, j, d2, diameters::Vector{Float64}, output::EnergyAndForces, pot::T
) where {T<:Potential}
    d = sqrt(d2)
    r = x - y
    (uij, fij) = evaluate(pot, d, diameters[i], diameters[j])
    sumies = @. fij * r / d
    output.virial += dot(sumies, r)
    output.energy += uij
    output.forces[i] .+= sumies
    output.forces[j] .-= sumies

    return output
end
