struct Parameters
    ρ::Float64
    n_particles::Int
    dt::Float64
end

struct SimulationState{T,U,V,W}
    # This field contains the cell lists for the system itself
    system::T
    # The array that contains the diameters of the particles
    diameters::U
    # The RNG
    rng::V
    # The size of the simulation box
    boxl::Float64
    # The container for the velocities
    velocities::Vector{SVector{2,Float64}}
    # The images for the particles
    images::Vector{W}
    # The dimension of the system
    dimension::Int
    # The degrees of freedom
    nf::Float64
end

abstract type Ensemble end

struct NVT <: Ensemble
    # Target temperature
    ktemp::Float64
    # Damping constant
    tau::Float64
end

struct Brownian <: Ensemble
    # Target temperature
    ktemp::Float64
end

struct NVE <: Ensemble end
