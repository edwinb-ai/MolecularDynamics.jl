"""
    Potential

Abstract supertype of all pair potentials. A subtype must implement
`evaluate(pot, r, sigma1, sigma2) -> (energy, force)`, where `force = -dU/dr`.
It may also implement [`evaluate_r2`](@ref) for a faster, `sqrt`-free path.
"""
abstract type Potential end

# Interface function that every potential should implement.
function evaluate(pot::Potential, r::Real; kwargs...)
    return error("evaluate not implemented for potential type: $(typeof(pot))")
end

function evaluate(pot::Potential, r, sigma1, sigma2)
    return error("evaluate not implemented for potential type: $(typeof(pot))")
end

"""
    evaluate_r2(pot, r2, sigma1, sigma2) -> (energy, force / r)

Evaluate a pair interaction from the squared distance `r2`. Returns the energy and the
force magnitude divided by the distance, which is what the force loop needs. The default
falls back to [`evaluate`](@ref); overload it to avoid the `sqrt` and division.
"""
@inline function evaluate_r2(pot::Potential, r2, sigma1, sigma2)
    r = sqrt(r2)
    (uij, fij) = evaluate(pot, r, sigma1, sigma2)
    return uij, fij / r
end

"""
    Parameters(ρ, n_particles, dt, potential)

Number density, number of particles, time step and pair potential of a simulation.
"""
struct Parameters{P<:Potential,T<:AbstractFloat,N<:Integer}
    ρ::T
    n_particles::N
    dt::T
    potential::P
end

"""
    SimulationState

Mutable state of a simulation. Positions are only wrapped into the box when the
neighbor list is rebuilt, so between rebuilds they may lie slightly outside it; use
[`wrapped_positions`](@ref) to obtain wrapped coordinates and image counters.
"""
mutable struct SimulationState{D,T<:AbstractFloat,M,R,NL}
    # Particle positions (unwrapped since the last neighbor list build)
    positions::Vector{SVector{D,T}}
    # Particle velocities, set with `initialize_velocities`
    velocities::Vector{SVector{D,T}}
    # Forces from the last force evaluation
    forces::Vector{SVector{D,T}}
    # Periodic image counters for unwrapping
    images::Vector{SVector{D,Int32}}
    # The array that contains the diameters of the particles
    diameters::Vector{T}
    # The simulation box, columns are the lattice vectors
    unitcell::M
    # The RNG
    rng::R
    # Verlet neighbor list
    neighbors::NL
    # The dimension of the system
    dimension::Int
    # The degrees of freedom
    nf::T
    # Potential energy and virial from the last force evaluation
    energy::T
    virial::T
end

abstract type Ensemble end

"""
    NVT(ktemp, tau)

Canonical ensemble with the Bussi stochastic velocity-rescaling thermostat. `ktemp` is
either a number or a callable `step -> temperature` (e.g. a [`LinearRamp`](@ref)), and
`tau` the thermostat time constant.
"""
struct NVT{U,T<:AbstractFloat} <: Ensemble
    # Target temperature
    ktemp::U
    # Damping constant
    tau::T
end

# For backward compatibility, allow construction with a constant value:
NVT(ktemp::T, tau::T) where {T<:AbstractFloat} = NVT(step -> ktemp, tau)

"""
    Brownian(ktemp)

Overdamped Brownian dynamics at temperature `ktemp`, with unit diffusion coefficient.
"""
struct Brownian{T<:AbstractFloat} <: Ensemble
    # Target temperature
    ktemp::T
end

"""
    NVE()

Microcanonical ensemble (plain velocity Verlet).
"""
struct NVE <: Ensemble end
