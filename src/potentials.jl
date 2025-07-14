# Some numerical constants
const b_param = 1.0204081632653061
const a_param = 134.5526623421209

struct PseudoHS{F<:Function} <: Potential
    potf::F
end

PseudoHS() = PseudoHS(pseudohs)

function evaluate(pot::PseudoHS, r::Float64, sigma1::Float64, sigma2::Float64)
    sigma = (sigma1 + sigma2) / 2.0
    return pot.potf(r, sigma; lambda=50.0)
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

"""
    struct LennardJones

Standard Lennard-Jones potential, with optional energy and force shifting.
- `ϵ`: Well depth parameter.
- `σ`: Size parameter.
- `r_cut`: Potential cutoff radius.
- `shift`: If true, applies energy shift so that V(r_cut) = 0.
- `force_shift`: If true, applies force shift so that both V(r_cut) = 0 and F(r_cut) = 0.
"""
struct LennardJones <: Potential
    epsilon::Float64
    sigma::Float64
    r_cut::Float64
    shift::Bool
    force_shift::Bool
    tail_correction::Bool
    V_cut::Float64
    F_cut::Float64
end

function LennardJones(;
    epsilon=1.0, sigma=1.0, r_cut=2.5, shift=false, force_shift=false, tail_correction=false
)
    srcut = sigma / r_cut
    srcut2 = srcut^2
    srcut6 = srcut2^3
    srcut12 = srcut6^2
    Vcut = 4.0 * epsilon * (srcut12 - srcut6)
    Fcut = 24.0 * epsilon * (2.0 * srcut12 - srcut6) / r_cut
    return LennardJones(
        epsilon, sigma, r_cut, shift, force_shift, tail_correction, Vcut, Fcut
    )
end

@inline function lj_unshifted(r, epsilon, sigma, r_cut)
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

@inline function lj_force_shifted(r, epsilon, sigma, r_cut, Vcut, Fcut)
    if r >= r_cut
        return 0.0, 0.0
    end
    sr = sigma / r
    sr2 = sr^2
    sr6 = sr2^3
    sr12 = sr6^2
    V = 4.0 * epsilon * (sr12 - sr6) - Vcut - (r - r_cut) * Fcut
    F = 24.0 * epsilon * (2.0 * sr12 - sr6) / r - Fcut
    return V, F
end

"""
    ener_lrc(cutoff, density, sigma=1.0)

Compute the standard long-range energy correction for Lennard-Jones with a sharp cutoff.
Returns the *total* energy correction for the system.
"""
@inline function ener_lrc(cutoff, density, sigma=1.0)
    uij = (((sigma / cutoff)^9) / 3.0) - ((sigma / cutoff)^3)
    uij *= 8.0 * pi * density / 3.0
    return uij
end

"""
    pressure_lrc(cutoff, density, sigma=1.0)

Compute the standard long-range pressure correction for Lennard-Jones with a sharp cutoff.
Returns the *total* pressure correction for the system.
"""
@inline function pressure_lrc(cutoff, density, sigma=1.0)
    sr3 = (sigma / cutoff)^3
    result = (2.0 * sr3^3 / 3.0) - sr3
    result *= 16.0 * pi * density^2 / 3.0
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
    return pot.tail_correction ? ener_lrc(pot.r_cut, ρ, pot.sigma) * N : 0.0
end

"""
    pressure_lrc(pot::LennardJones, N, V)

Return the analytic long-range pressure correction for `LennardJones` potential if enabled,
otherwise returns 0.0.
"""
@inline function pressure_lrc(pot::LennardJones, N, V)
    ρ = N / V
    return pot.tail_correction ? pressure_lrc(pot.r_cut, ρ, pot.sigma) : 0.0
end

"""
    evaluate(pot::LennardJones, r::Real; sigma1=pot.σ, sigma2=pot.σ)

Evaluate the Lennard-Jones potential for a given distance `r`, with optional individual sigmas.
Returns a tuple `(energy, force)`.
"""
@inline function evaluate(pot::LennardJones, r::Float64, sigma1::Float64, sigma2::Float64)
    # ! FIXME: Mixing rules cannot be assumed for the user
    σ = (sigma1 + sigma2) / 2.0
    return lj_unshifted(r, pot.epsilon, σ, pot.r_cut)
end

"""
    struct LennardJonesXPLOR

Lennard-Jones potential with XPLOR smooth cutoff and optional long-range corrections.
- `ϵ`: Well depth parameter.
- `σ`: Size parameter.
- `r_on`: Switching function start radius.
- `r_cut`: Potential cutoff radius.
- `tail_correction`: If true, applies long-range (tail) corrections to energy and pressure.
"""
struct LennardJonesXPLOR <: Potential
    ϵ::Float64
    σ::Float64
    r_on::Float64
    r_cut::Float64
    tail_correction::Bool
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
        dnum1 =
            -4.0 * r * (rc2 - r2) * (rc2 + 2.0 * r2 - 3.0 * ron2) +
            2.0 * (rc2 - r2) * 2.0 * r * (rc2 + 2.0 * r2 - 3.0 * ron2) +
            (rc2 - r2)^2 * 4.0 * r
        dS = dnum1 / denom
        return S, dS
    else
        return 0.0, 0.0
    end
end

"""
    lj_xplor(r, lj::LennardJonesXPLOR)

Compute the Lennard-Jones XPLOR-shifted potential and force for distance `r`.
Returns a tuple `(energy, force)`.
"""
FastPow.@fastpow function lj_xplor(r, lj::LennardJonesXPLOR)
    if r >= lj.r_cut
        return 0.0, 0.0
    end

    σ, ϵ = lj.σ, lj.ϵ
    sr = σ / r
    sr2 = sr^2
    sr6 = sr2^3
    sr12 = sr6^2

    V = 4.0 * ϵ * (sr12 - sr6)
    F = 24.0 * ϵ * (2.0 * sr12 - sr6) / r

    S, dS = xplor_switch(r, lj.r_on, lj.r_cut)
    # The force is: d/dr [V(r) * S(r)] = S(r) * F(r) + V(r) * dS/dr
    force = S * F + V * dS

    return V * S, force
end

"""
    evaluate(pot::LennardJonesXPLOR, r::Real; sigma1=pot.σ, sigma2=pot.σ)

Evaluate the Lennard-Jones XPLOR potential for a given distance `r`, with optional individual sigmas.
Returns a tuple `(energy, force)`.
"""
function evaluate(pot::LennardJonesXPLOR, r::Real; sigma1=pot.σ, sigma2=pot.σ)
    # Use arithmetic mean for cross-interactions (standard Lorentz-Berthelot)
    σ = (sigma1 + sigma2) / 2.0
    scaled_pot = LennardJonesXPLOR(pot.ϵ, σ, pot.r_on, pot.r_cut, pot.tail_correction)
    return lj_xplor(r, scaled_pot)
end

"""
    lj_xplor_tail_energy(N, V, pot::LennardJonesXPLOR)

Compute the analytic long-range energy correction for Lennard-Jones XPLOR potential.
"""
function lj_xplor_tail_energy(N, V, pot::LennardJonesXPLOR)
    ρ = N / V
    σ, ϵ, rc = pot.σ, pot.ϵ, pot.r_cut
    return (8.0 / 3.0) * pi * ρ * N * ϵ * σ^3 * ((1.0 / 3.0) * (σ / rc)^9 - (σ / rc)^3)
end

"""
    lj_xplor_tail_pressure(N, V, pot::LennardJonesXPLOR)

Compute the analytic long-range pressure correction for Lennard-Jones XPLOR potential.
"""
function lj_xplor_tail_pressure(N, V, pot::LennardJonesXPLOR)
    ρ = N / V
    σ, ϵ, rc = pot.σ, pot.ϵ, pot.r_cut
    return (16.0 / 3.0) * pi * ρ^2 * ϵ * σ^3 * ((2.0 / 3.0) * (σ / rc)^9 - (σ / rc)^3)
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

"""
    energy_lrc(pot::LennardJonesXPLOR, N, V)

Return the analytic long-range energy correction for `LennardJonesXPLOR` potential if enabled,
otherwise returns 0.0.
"""
function energy_lrc(pot::LennardJonesXPLOR, N, V)
    return pot.tail_correction ? lj_xplor_tail_energy(N, V, pot) : 0.0
end

"""
    pressure_lrc(pot::LennardJonesXPLOR, N, V)

Return the analytic long-range pressure correction for `LennardJonesXPLOR` potential if enabled,
otherwise returns 0.0.
"""
function pressure_lrc(pot::LennardJonesXPLOR, N, V)
    return pot.tail_correction ? lj_xplor_tail_pressure(N, V, pot) : 0.0
end
