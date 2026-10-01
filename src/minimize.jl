"""
    fire_minimize!(state, params; kwargs...) -> (energy, converged)

Minimize the potential energy in place with the Fast Inertial Relaxation Engine (FIRE).
Converges when the root mean square force per degree of freedom drops below `tol`.

# Keyword arguments
- `max_steps=10000`: maximum number of FIRE steps.
- `tol=1e-6`: convergence tolerance for the root mean square force.
- `dt_initial=0.01`, `dt_max=0.1`: initial and maximum time step.
- `alpha0=0.1`: initial velocity mixing parameter.
- `f_inc=1.2`, `f_dec=0.2`: time step increase and decrease factors.
- `Nmin=5`: number of downhill steps before the time step may grow.
"""
function fire_minimize!(
    state::SimulationState,
    params::Parameters;
    max_steps::Int=10000,
    tol::Float64=1e-6,
    dt_initial::Float64=0.01,
    dt_max::Float64=0.1,
    alpha0::Float64=0.1,
    f_inc::Float64=1.2,
    f_dec::Float64=0.2,
    Nmin::Int=5,
)
    # Extract information from the state and parameters
    positions = state.positions
    forces = state.forces
    images = state.images
    neighbors = state.neighbors
    potential = params.potential
    N = length(positions)

    # Initialize internal variables
    α = alpha0
    steps_since_neg = 0
    dt = dt_initial
    # The array that holds the information of the velocities
    v = zero(forces)
    # Degrees of freedom
    ndof = state.dimension * (N - 1.0)

    build!(neighbors, positions, images)
    for step in 1:max_steps
        update!(neighbors, positions, images)
        compute_forces!(state, potential)
        energy = state.energy

        F_norm = sqrt(sum(f -> sum(abs2, f), forces))

        if step % 100 == 0
            print_fnorm = F_norm / sqrt(ndof)
            print_energy = energy / N
            @info "Step $(step): F_rms = $(print_fnorm), energy = $(print_energy)"
        end

        if F_norm / sqrt(ndof) < tol
            build!(neighbors, positions, images)
            return energy, true
        end

        @inbounds for i in eachindex(v, forces)
            v[i] += dt * forces[i]
        end

        P = sum(i -> dot(v[i], forces[i]), eachindex(v, forces))

        v_norm = sqrt(sum(vi -> sum(abs2, vi), v))
        if v_norm > 0 && F_norm > 0
            scale = α * (v_norm / F_norm)
            @inbounds for i in eachindex(v, forces)
                v[i] = (1.0 - α) * v[i] + scale * forces[i]
            end
        end

        if P > 0
            steps_since_neg += 1
            if steps_since_neg > Nmin
                dt = min(dt * f_inc, dt_max)
                α *= 0.99
            end
        else
            dt = max(dt * f_dec, dt_initial)
            fill!(v, zero(eltype(v)))
            α = alpha0
            steps_since_neg = 0
        end

        @inbounds for i in eachindex(positions, v)
            positions[i] += dt * v[i]
        end
    end

    update!(neighbors, positions, images)
    compute_forces!(state, potential)
    build!(neighbors, positions, images)
    F_norm = sqrt(sum(f -> sum(abs2, f), forces)) / sqrt(ndof)

    @warn "FIRE did not converge after $(max_steps) steps; final F_norm = $(F_norm)"

    return state.energy, false
end

"""
    minimize!(state, params, pathname, dimension; method=:FIRE, save_config="minimized.xyz", kwargs...)

Minimize the energy of `state` in place and append the result to `pathname/save_config`.
Only `method=:FIRE` is available; `kwargs` are forwarded to [`fire_minimize!`](@ref).
Returns `(energy, converged)`.
"""
function minimize!(
    state::SimulationState,
    params::Parameters,
    pathname::String,
    dimension::Int;
    method::Symbol=:FIRE,
    save_config::String="minimized.xyz",
    kwargs...,
)
    if method == :FIRE
        result = fire_minimize!(state, params; kwargs...)
    else
        error("Unknown minimization method: $method")
    end

    # Unpack some variables and save the final configuration to file
    unitcell = state.unitcell
    positions = state.positions
    diameters = state.diameters
    n_particles = params.n_particles
    write_to_file(
        joinpath(pathname, save_config),
        0,
        unitcell,
        n_particles,
        positions,
        diameters,
        dimension,
    )

    return result
end
