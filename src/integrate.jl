const sqthree = sqrt(3.0)

function register_images_and_wrap!(x, image, boxl)
    # Compute number of box crossings for each dimension
    n_cross = floor.(x ./ boxl)

    # Update the image vector
    @. image += Int(n_cross)

    # Wrap the position
    wrapped_x = @. x - boxl * n_cross

    return wrapped_x
end

function integrate_half!(positions, images, velocities, forces, dt, boxl)
    for i in eachindex(positions, forces, velocities)
        f = forces[i]
        x = positions[i]
        v = velocities[i]
        # ! Important: There is a mass in the force term
        velocities[i] = @. v + (f * dt / 2.0)
        positions[i] = @. x + (velocities[i] * dt)
        positions[i] = register_images_and_wrap!(positions[i], images[i], boxl)
    end

    return nothing
end

function integrate_second_half!(velocities, forces, dt)
    for i in eachindex(velocities, forces)
        f = forces[i]
        v = velocities[i]
        velocities[i] = @. v + (f * dt / 2.0)
    end

    return nothing
end

function ensemble_step!(::NVE, velocities, params::Parameters, state::SimulationState, step::Int)
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
