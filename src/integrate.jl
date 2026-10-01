const sqthree = sqrt(3.0)

"""
    integrate_half!(positions, velocities, forces, dt)

Velocity Verlet first half-step: half kick of the velocities, then a full drift of the
positions. Positions are not wrapped; that happens when the neighbor list is rebuilt.
"""
function integrate_half!(positions, velocities, forces, dt)
    half_dt = dt / 2.0
    @inbounds for i in eachindex(positions, velocities, forces)
        velocities[i] += forces[i] * half_dt
        positions[i] += velocities[i] * dt
    end

    return nothing
end

"""
    integrate_second_half!(velocities, forces, dt)

Velocity Verlet second half-step: half kick with the new forces.
"""
function integrate_second_half!(velocities, forces, dt)
    half_dt = dt / 2.0
    @inbounds for i in eachindex(velocities, forces)
        velocities[i] += forces[i] * half_dt
    end

    return nothing
end

"""
    ensemble_step!(ensemble, velocities, params, state, step) -> temperature

Apply the thermostat of `ensemble` (if any) and return the instantaneous temperature.
"""
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

"""
    sample_uniform(rng, ::Type{SVector{D,T}})

Uniform random vector with zero mean and unit variance per component.
"""
@inline function sample_uniform(rng, ::Type{SVector{D,T}}) where {D,T}
    return (2.0 * rand(rng, SVector{D,T}) .- 1.0) * sqthree
end

"""
    integrate_brownian!(positions, forces, dt, rng, ktemp, sigma)

Euler-Maruyama step of overdamped Brownian dynamics with unit diffusion coefficient:
`x += F dt / kT + sigma ξ`, with `ξ` uniform noise of unit variance and
`sigma = sqrt(2 dt)`. Runs serially so that the shared `rng` stays reproducible.
"""
function integrate_brownian!(positions, forces, dt, rng, ktemp, sigma)
    @inbounds for i in eachindex(positions, forces)
        noise = sample_uniform(rng, eltype(positions))
        positions[i] += forces[i] * (dt / ktemp) + noise * sigma
    end

    return nothing
end
