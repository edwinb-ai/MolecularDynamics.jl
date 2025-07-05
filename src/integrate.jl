const sqthree = sqrt(3.0)

"""
    WrappedBoxSolver{N}

Caches the static box matrix and its inverse for efficient wrapping and image registration.
"""
struct WrappedBoxSolver{N}
    boxmat::SMatrix{N,N,Float64}
    boxinv::SMatrix{N,N,Float64}
end

function WrappedBoxSolver(boxmat::SMatrix{N,N,Float64}) where {N}
    boxinv = inv(boxmat)
    return WrappedBoxSolver{N}(boxmat, boxinv)
end

"""
    register_images_and_wrap!(x, image, solver::WrappedBoxSolver{N})

Efficient, robust periodic wrap for general box using StaticArrays and cached inverse.
- `x`: SVector{N,Float64}, particle position
- `image`: MVector{N,Int}, image vector (in-place update)
- `solver`: WrappedBoxSolver
Returns: SVector{N,Float64}, wrapped position
"""
function register_images_and_wrap!(
    x::SVector{N,Float64}, image::MVector{N,T}, solver::WrappedBoxSolver{N}
) where {N,T<:Integer}
    # Compute fractional coordinates (boxinv * x is fast and type-stable)
    x_frac = solver.boxinv * x
    n_cross = floor.(x_frac)
    @. image += Int(n_cross)
    x_frac_wrapped = x_frac .- n_cross
    wrapped_x = solver.boxmat * x_frac_wrapped
    return wrapped_x
end

function integrate_half!(positions, images, velocities, forces, dt, boxl)
    @threads for i in eachindex(positions, forces, velocities)
        f = forces[i]
        x = positions[i]
        v = velocities[i]
        # ! Important: There should be a mass in the force term
        velocities[i] = @. v + (f * dt / 2.0)
        positions[i] = @. x + (velocities[i] * dt)
        positions[i] = register_images_and_wrap!(positions[i], images[i], boxl)
    end

    return nothing
end

function integrate_second_half!(velocities, forces, dt)
    @threads for i in eachindex(velocities, forces)
        f = forces[i]
        v = velocities[i]
        velocities[i] = @. v + (f * dt / 2.0)
    end

    return nothing
end

function ensemble_step!(
    ::NVE, velocities, params::Parameters, state::SimulationState, step::Int
)
    return compute_temperature(velocities, state.nf)
end

function ensemble_step!(
    ensemble::NVT, velocities, params::Parameters, state::SimulationState, step::Int
)
    temperature = ensemble.ktemp(step)
    # Apply thermostat, e.g., Bussi thermostat
    bussi!(velocities, temperature, state.nf, params.dt, ensemble.tau, state.rng)
    return compute_temperature(velocities, state.nf)
end

@inline function sample_uniform!(vector, rng)
    rand!(rng, vector)
    @. vector = (2.0 * vector - 1.0) * sqthree

    return nothing
end

function integrate_brownian!(
    positions, images, forces, dt, boxl, rng, dimension, ktemp, sigma
)
    noise = zeros(MVector{dimension,Float64})

    @inbounds for i in eachindex(positions, forces)
        f = forces[i]
        x = positions[i]
        sample_uniform!(noise, rng)
        positions[i] = @. x + (f * dt / ktemp) + (noise * sigma)
        positions[i] = register_images_and_wrap!(positions[i], images[i], boxl)
    end

    return nothing
end
