# Some numerical constants
const b_param = 1.0204081632653061
const a_param = 134.5526623421209

struct PseudoHS{F<:Function} <: Potential
    potf::F
end

PseudoHS() = PseudoHS(pseudohs)

function evaluate(pot::PseudoHS, r::Real; sigma1=1.0, sigma2=1.0, lambda=50.0)
    sigma = (sigma1 + sigma2) / 2.0
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
end

function LennardJones(;
    epsilon::Float64=1.0,
    sigma::Float64=1.0,
    r_cut::Float64=2.5,
    shift::Bool=false,
    force_shift::Bool=false,
    tail_correction::Bool=false,
)
    return LennardJones(epsilon, sigma, r_cut, shift, force_shift, tail_correction)
end

"""
    lj_energy_force(r, pot::LennardJones)

Compute the (possibly shifted) Lennard-Jones potential and force for distance `r`.
Returns a tuple `(energy, force)`.
"""
FastPow.@fastpow function lj_energy_force(r, pot::LennardJones)
    σ, ϵ, r_cut = pot.sigma, pot.epsilon, pot.r_cut
    if r >= r_cut
        return 0.0, 0.0
    end
    sr = σ / r
    sr2 = sr^2
    sr6 = sr2^3
    sr12 = sr6^2
    V = 4.0 * ϵ * (sr12 - sr6)
    F = 24.0 * ϵ * (2.0 * sr12 - sr6) / r

    if pot.force_shift
        # Linear force shift (force and energy go to zero at r_cut)
        srcut = σ / r_cut
        srcut2 = srcut^2
        srcut6 = srcut2^3
        srcut12 = srcut6^2
        Vcut = 4.0 * ϵ * (srcut12 - srcut6)
        Fcut = 24.0 * ϵ * (2 * srcut12 - srcut6) / r_cut
        V = V - Vcut - (r - r_cut) * Fcut
        F = F - Fcut
    elseif pot.shift
        # Energy shift (energy only goes to zero at r_cut)
        srcut = σ / r_cut
        srcut2 = srcut^2
        srcut6 = srcut2^3
        srcut12 = srcut6^2
        Vcut = 4.0 * ϵ * (srcut12 - srcut6)
        V = V - Vcut
        # F is unchanged
    end

    return V, F
end

"""
    ener_lrc(cutoff, density, sigma=1.0)

Compute the standard long-range energy correction for Lennard-Jones with a sharp cutoff.
Returns the *total* energy correction for the system.
"""
function ener_lrc(cutoff, density, sigma=1.0)
    uij = (((sigma / cutoff)^9) / 3.0) - ((sigma / cutoff)^3)
    uij *= 8.0 * pi * density / 3.0
    return uij
end

"""
    pressure_lrc(cutoff, density, sigma=1.0)

Compute the standard long-range pressure correction for Lennard-Jones with a sharp cutoff.
Returns the *total* pressure correction for the system.
"""
function pressure_lrc(cutoff, density, sigma=1.0)
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
function energy_lrc(pot::LennardJones, N, V)
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
function pressure_lrc(pot::LennardJones, N, V)
    ρ = N / V
    return pot.tail_correction ? pressure_lrc(pot.r_cut, ρ, pot.sigma) : 0.0
end

"""
    evaluate(pot::LennardJones, r::Real; sigma1=pot.σ, sigma2=pot.σ)

Evaluate the Lennard-Jones potential for a given distance `r`, with optional individual sigmas.
Returns a tuple `(energy, force)`.
"""
function evaluate(pot::LennardJones, r::Real; sigma1=pot.sigma, sigma2=pot.sigma)
    # ! FIXME: Mixing rules cannot be assumed for the user
    σ = (sigma1 + sigma2) / 2.0
    scaled_pot = LennardJones(;
        epsilon=pot.epsilon,
        sigma=σ,
        r_cut=pot.r_cut,
        shift=pot.shift,
        force_shift=pot.force_shift,
        tail_correction=pot.tail_correction,
    )
    return lj_energy_force(r, scaled_pot)
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
