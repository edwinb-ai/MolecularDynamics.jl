const sqthree = sqrt(3.0)

"""
    integrate_half!(positions, images, velocities, forces, dt, unitcell, unitcell_inv)

Velocity Verlet first half-step using generic unitcell.
"""
function integrate_half!(positions, images, velocities, forces, dt, unitcell, unitcell_inv)
    @threads for i in eachindex(positions, forces, velocities)
        @inbounds begin
            f = forces[i]
            x = positions[i]
            v = velocities[i]
            velocities[i] = @. v + (f * dt / 2.0)
            positions[i] = @. x + (velocities[i] * dt)
            positions[i] = wrap_to_box!(positions[i], images[i], unitcell, unitcell_inv)
        end
    end

    return nothing
end

"""
    integrate_second_half!(velocities, forces, dt)

Velocity Verlet second half-step.
"""
function integrate_second_half!(velocities, forces, dt)
    @threads for i in eachindex(velocities, forces)
        @inbounds begin
            v = velocities[i]
            f = forces[i]
            velocities[i] = @. v + (f * dt / 2.0)
        end
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

"""
    integrate_brownian!(positions, images, forces, dt, unitcell, unitcell_inv, rng, dimension, ktemp, sigma)

Brownian dynamics integrator using generic unitcell.
"""
function integrate_brownian!(
    positions, images, forces, dt, unitcell, unitcell_inv, rng, dimension, ktemp, sigma
)
    noise = zeros(MVector{dimension,Float64})

    @threads for i in eachindex(positions, forces)
        @inbounds begin 
            f = forces[i]
            x = positions[i]
            sample_uniform!(noise, rng)
            positions[i] = @. x + (f * dt / ktemp) + (noise * sigma)
            positions[i] = wrap_to_box!(positions[i], images[i], unitcell, unitcell_inv)
        end
    end

    return nothing
end
