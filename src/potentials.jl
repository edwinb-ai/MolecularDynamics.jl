# Some numerical constants
const b_param = 1.0204081632653061
const a_param = 134.5526623421209

"""
    PseudoHS()

Pseudo hard-sphere potential: a (50, 49) Mie potential, cut and shifted at its minimum.
"""
struct PseudoHS{F<:Function} <: Potential
    potf::F
end

PseudoHS() = PseudoHS(pseudohs)

"""
    evaluate(pot::PseudoHS, r, sigma1, sigma2)

Evaluate the pseudo hard-sphere potential with `σ = (sigma1 + sigma2) / 2`.
"""
function evaluate(pot::PseudoHS, r::Real, sigma1::Real, sigma2::Real)
    sigma = (sigma1 + sigma2) / 2.0
    return pot.potf(r, sigma; lambda=50.0)
end

"""
    pseudohs(rij, sigma; lambda=50.0) -> (energy, force)

Pseudo hard-sphere energy and force, non-zero for `rij < b_param * sigma`.
"""
FastPow.@fastpow function pseudohs(rij, sigma; lambda=50.0)
    uij = 0.0
    fij = 0.0

    if rij < b_param * sigma
        uij = a_param * ((sigma / rij)^lambda - (sigma / rij)^(lambda - 1.0))
        uij += 1.0
        fij = lambda * (sigma / rij)^lambda
        fij -= (lambda - 1.0) * (sigma / rij)^(lambda - 1.0)
        fij *= a_param / rij
    end

    return uij, fij
end

"""
    LennardJones(; epsilon=1.0, sigma=1.0, r_cut=2.5, shift=false, force_shift=false, tail_correction=false)

Lennard-Jones potential truncated at `r_cut`, with optional energy and force shifting.
The size of each pair is the mean of the two particle diameters.
- `epsilon`: well depth.
- `sigma`: size used for the tail corrections and the stored `V_cut`, `F_cut`.
- `shift`: shift the energy so that V(r_cut) = 0.
- `force_shift`: shift energy and force so that V(r_cut) = 0 and F(r_cut) = 0.
- `tail_correction`: add the analytic long-range corrections of the unshifted potential.
"""
struct LennardJones{T<:AbstractFloat} <: Potential
    epsilon::T
    sigma::T
    r_cut::T
    shift::Bool
    force_shift::Bool
    tail_correction::Bool
    V_cut::T
    F_cut::T
end

function LennardJones(;
    epsilon=1.0, sigma=1.0, r_cut=2.5, shift=false, force_shift=false, tail_correction=false
)
    (Vcut, Fcut) = lj_cut_values(epsilon, sigma, r_cut)
    return LennardJones(
        epsilon, sigma, r_cut, shift, force_shift, tail_correction, Vcut, Fcut
    )
end

"""
    lj_cut_values(epsilon, sigma, r_cut) -> (V_cut, F_cut)

Lennard-Jones energy and force at the cutoff.
"""
@inline function lj_cut_values(epsilon, sigma, r_cut)
    srcut = sigma / r_cut
    srcut2 = srcut * srcut
    srcut6 = srcut2 * srcut2 * srcut2
    srcut12 = srcut6 * srcut6
    Vcut = 4.0 * epsilon * (srcut12 - srcut6)
    Fcut = 24.0 * epsilon * (2.0 * srcut12 - srcut6) / r_cut
    return Vcut, Fcut
end

"""
    lj_unshifted(r, epsilon, sigma, r_cut) -> (energy, force)

Plain Lennard-Jones energy and force, truncated at `r_cut`.
"""
FastPow.@fastpow function lj_unshifted(r, epsilon, sigma, r_cut)
    if r >= r_cut
        return 0.0, 0.0
    end
    sr = sigma / r
    sr2 = sr^2
    sr6 = sr2^3
    sr12 = sr6^2
    V = 4.0 * epsilon * (sr12 - sr6)
    F = 24.0 * epsilon * (2.0 * sr12 - sr6) / r
    return V, F
end

"""
    lj_energy_shifted(r, epsilon, sigma, r_cut, Vcut) -> (energy, force)

Lennard-Jones with the energy shifted so that it vanishes at `r_cut`.
"""
@inline function lj_energy_shifted(r, epsilon, sigma, r_cut, Vcut)
    if r >= r_cut
        return 0.0, 0.0
    end
    sr = sigma / r
    sr2 = sr^2
    sr6 = sr2^3
    sr12 = sr6^2
    V = 4.0 * epsilon * (sr12 - sr6) - Vcut
    F = 24.0 * epsilon * (2.0 * sr12 - sr6) / r
    return V, F
end

"""
    lj_force_shifted(r, epsilon, sigma, r_cut, Vcut, Fcut) -> (energy, force)

Lennard-Jones with energy and force shifted so that both vanish at `r_cut`.
"""
FastPow.@fastpow function lj_force_shifted(r, epsilon, sigma, r_cut, Vcut, Fcut)
    if r >= r_cut
        return 0.0, 0.0
    end
    sr = sigma / r
    sr2 = sr^2
    sr6 = sr2^3
    sr12 = sr6^2
    V = 4.0 * epsilon * (sr12 - sr6) - Vcut + (r - r_cut) * Fcut
    F = 24.0 * epsilon * (2.0 * sr12 - sr6) / r - Fcut
    return V, F
end

"""
    ener_lrc(cutoff, density, sigma=1.0, epsilon=1.0)

Standard long-range energy correction per particle for Lennard-Jones with a sharp cutoff,
`(8π/3) ρ ε σ³ [(σ/rc)⁹/3 - (σ/rc)³]`.
"""
FastPow.@fastpow function ener_lrc(cutoff, density, sigma=1.0, epsilon=1.0)
    uij = (((sigma / cutoff)^9) / 3.0) - ((sigma / cutoff)^3)
    uij *= 8.0 * pi * density * epsilon * sigma^3 / 3.0
    return uij
end

"""
    pressure_lrc(cutoff, density, sigma=1.0, epsilon=1.0)

Standard long-range pressure correction for Lennard-Jones with a sharp cutoff,
`(16π/3) ρ² ε σ³ [2(σ/rc)⁹/3 - (σ/rc)³]`.
"""
FastPow.@fastpow function pressure_lrc(cutoff, density, sigma=1.0, epsilon=1.0)
    sr3 = (sigma / cutoff)^3
    result = (2.0 * sr3^3 / 3.0) - sr3
    result *= 16.0 * pi * density^2 * epsilon * sigma^3 / 3.0
    return result
end

"""
    energy_lrc(pot::LennardJones, N, V)

Return the analytic long-range energy correction for `LennardJones` potential if enabled,
otherwise returns 0.0.
"""
@inline function energy_lrc(pot::LennardJones, N, V)
    # By convention, use energy shift for plain and energy-shifted, and LRC if requested.
    # (Add a field if you want to toggle LRC on/off)
    ρ = N / V
    return pot.tail_correction ? ener_lrc(pot.r_cut, ρ, pot.sigma, pot.epsilon) * N : 0.0
end

"""
    pressure_lrc(pot::LennardJones, N, V)

Return the analytic long-range pressure correction for `LennardJones` potential if enabled,
otherwise returns 0.0.
"""
@inline function pressure_lrc(pot::LennardJones, N, V)
    ρ = N / V
    return pot.tail_correction ? pressure_lrc(pot.r_cut, ρ, pot.sigma, pot.epsilon) : 0.0
end

"""
    evaluate(pot::LennardJones, r, sigma1, sigma2)

Evaluate the Lennard-Jones potential at distance `r`, with `σ = (sigma1 + sigma2) / 2`,
applying the shifts selected in `pot`. Returns a tuple `(energy, force)`.
"""
function evaluate(pot::LennardJones, r::Real, sigma1::Real, sigma2::Real)
    # ! FIXME: Mixing rules cannot be assumed for the user
    σ = (sigma1 + sigma2) / 2.0
    if pot.force_shift
        (Vcut, Fcut) = lj_cut_values(pot.epsilon, σ, pot.r_cut)
        return lj_force_shifted(r, pot.epsilon, σ, pot.r_cut, Vcut, Fcut)
    elseif pot.shift
        (Vcut, _) = lj_cut_values(pot.epsilon, σ, pot.r_cut)
        return lj_energy_shifted(r, pot.epsilon, σ, pot.r_cut, Vcut)
    end
    return lj_unshifted(r, pot.epsilon, σ, pot.r_cut)
end

"""
    evaluate_r2(pot::LennardJones, r2, sigma1, sigma2)

Lennard-Jones evaluation from `r2`, equivalent to `evaluate` and `sqrt`-free unless
`force_shift` is set. Returns `(energy, force / r)`.
"""
@inline function evaluate_r2(
    pot::LennardJones, r2::Float64, sigma1::Float64, sigma2::Float64
)
    if r2 >= pot.r_cut^2
        return 0.0, 0.0
    end
    # The force shift depends on r itself
    if pot.force_shift
        r = sqrt(r2)
        (V, F) = evaluate(pot, r, sigma1, sigma2)
        return V, F / r
    end
    σ = (sigma1 + sigma2) / 2.0
    sr2 = σ * σ / r2
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6
    V = 4.0 * pot.epsilon * (sr12 - sr6)
    F_over_r = 24.0 * pot.epsilon * (2.0 * sr12 - sr6) / r2
    if pot.shift
        V -= first(lj_cut_values(pot.epsilon, σ, pot.r_cut))
    end
    return V, F_over_r
end

"""
    xplor_switch(r, r_on, r_cut)

Compute the value and derivative of the XPLOR switching function at distance `r`.
Returns `(S, dSdr)`.
"""
function xplor_switch(r, r_on, r_cut)
    if r < r_on
        return 1.0, 0.0
    elseif r < r_cut
        rc2, r2, ron2 = r_cut^2, r^2, r_on^2
        denom = (rc2 - ron2)^3
        num1 = (rc2 - r2)^2 * (rc2 + 2.0 * r2 - 3.0 * ron2)
        S = num1 / denom

        # Derivative dS/dr
        dS = -12.0 * r * (rc2 - r2) * (r2 - ron2) / denom
        return S, dS
    else
        return 0.0, 0.0
    end
end

# ----- Generic LRC interface for all potentials -----

"""
    energy_lrc(pot::Potential, N, V)

Generic interface for long-range energy correction. Returns 0.0 by default.
Override for potentials with analytic corrections.
"""
function energy_lrc(::Potential, N, V)
    return 0.0
end

"""
    pressure_lrc(pot::Potential, N, V)

Generic interface for long-range pressure correction. Returns 0.0 by default.
Override for potentials with analytic corrections.
"""
function pressure_lrc(::Potential, N, V)
    return 0.0
end
