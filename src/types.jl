abstract type Potential end

# Interface function that every potential should implement.
function evaluate(pot::Potential, r::Real; kwargs...)
    return error("evaluate not implemented for potential type: $(typeof(pot))")
end

struct Parameters{P<:Potential,T<:AbstractFloat,N<:Integer}
    ρ::T
    n_particles::N
    dt::T
    potential::P
end

mutable struct ParticleSystem{T,P,U}
    neighbor_system::T
    positions::P
    energy_forces::U
end

mutable struct SimulationState{T,W,M,F<:AbstractFloat,I<:Integer}
    # This field contains the cell lists for the system itself
    system::T
    # The array that contains the diameters of the particles
    diameters::Vector{Float64}
    # The RNG
    rng::AbstractRNG
    # The size of the simulation box
    unitcell::AbstractMatrix{Float64}
    # The container for the velocities
    velocities::W
    # The images for the particles
    images::M
    # The dimension of the system
    dimension::I
    # The degrees of freedom
    nf::F
end

abstract type Ensemble end

struct NVT{U,T<:AbstractFloat} <: Ensemble
    # Target temperature
    ktemp::U
    # Damping constant
    tau::T
end

# For backward compatibility, allow construction with a constant value:
NVT(ktemp::T, tau::T) where {T<:AbstractFloat} = NVT(step -> ktemp, tau)

struct Brownian{T<:AbstractFloat} <: Ensemble
    # Target temperature
    ktemp::T
end

struct NVE <: Ensemble end

mutable struct EnergyAndForces{T,U<:AbstractFloat}
    energy::U
    virial::U
    forces::T
end
