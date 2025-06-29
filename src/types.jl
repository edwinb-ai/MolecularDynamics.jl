abstract type Potential end

# Interface function that every potential should implement.
function evaluate(pot::Potential, r::Real; kwargs...)
    return error("evaluate not implemented for potential type: $(typeof(pot))")
end

struct Parameters
    ρ::Float64
    n_particles::Int
    dt::Float64
    potential::Potential
end

struct SimulationState{T,U,V,W,M}
    # This field contains the cell lists for the system itself
    system::T
    # The array that contains the diameters of the particles
    diameters::U
    # The RNG
    rng::V
    # The size of the simulation box
    boxl::Float64
    # The container for the velocities
    velocities::W
    # The images for the particles
    images::M
    # The dimension of the system
    dimension::Int
    # The degrees of freedom
    nf::Float64
end

abstract type Ensemble end

struct NVT{T} <: Ensemble
    # Target temperature
    ktemp::T
    # Damping constant
    tau::Float64
end

# For backward compatibility, allow construction with a constant value:
NVT(ktemp::Float64, tau::Float64) = NVT(step -> ktemp, tau)

struct Brownian <: Ensemble
    # Target temperature
    ktemp::Float64
end

struct NVE <: Ensemble end

mutable struct EnergyAndForces{T}
    energy::Float64
    virial::Float64
    forces::T
end
