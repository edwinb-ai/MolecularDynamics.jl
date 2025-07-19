abstract type Potential end

# Interface function that every potential should implement.
function evaluate(pot::Potential, r::Real; kwargs...)
    return error("evaluate not implemented for potential type: $(typeof(pot))")
end

mutable struct NeighborLists{T,U}
    neighbor_system::T
    cached_neighbor_list::Vector{Tuple{Int,Int,Float64}}
    last_rebuild_positions::U
end

struct Parameters
    ρ::Float64
    n_particles::Int
    dt::Float64
    potential::Potential
end

mutable struct SimulationState{T,W,M,N}
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
    dimension::Int
    # The degrees of freedom
    nf::Float64
    # The neighbor lists
    neighbor_list::N
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
