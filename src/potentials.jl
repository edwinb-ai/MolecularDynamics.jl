# Some numerical constants
const b_param = 1.0204081632653061
const a_param = 134.5526623421209

struct PseudoHS{F<:Function} <: Potential
    potf::F
end

PseudoHS() = PseudoHS(pseudohs)

function evaluate(pot::PseudoHS, r::Real, sigma::Real=1.0; lambda=50.0)
    return pot.potf(r, sigma; lambda=lambda)
end

FastPow.@fastpow function pseudohs(rij, sigma; lambda=50.0)
    uij = 0.0
    fij = 0.0

    if rij < b_param
        uij = a_param * ((sigma / rij)^lambda - (sigma / rij)^(lambda - 1.0))
        uij += 1.0
        fij = lambda * (sigma / rij)^(lambda + 1.0)
        fij -= (lambda - 1.0) * (sigma / rij)^lambda
        fij *= a_param
    end

    return uij, fij
end

FastPow.@fastpow function lj(rij, sigma=1.0)
    uij = (sigma / rij)^12 - (sigma / rij)^6
    uij *= 4.0
    fij = 24.0 * (2.0 * (sigma / rij)^13 - (sigma / rij)^7)

    return uij, fij
end

function ener_lrc(cutoff, density, sigma=1.0)
    uij = (((sigma / cutoff)^9) / 3.0) - ((sigma / cutoff)^3)
    uij *= 8.0 * pi * density / 3.0

    return uij
end

function pressure_lrc(cutoff, density, sigma=1.0)
    sr3 = (sigma / cutoff)^3
    result = (2.0 * sr3^3 / 3.0) - sr3
    result *= 16.0 * pi * density^2 / 3.0

    return result
end
