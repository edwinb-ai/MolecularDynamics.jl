"""
    fire_minimize!(state::SimulationState, params::Parameters; kwargs...)

Perform energy minimization using the Fast Inertial Relaxation Engine (FIRE) algorithm.

# Arguments
- `state::SimulationState`: The current simulation state containing positions, system, etc. The minimizer will update positions in-place.
- `params::Parameters`: Simulation parameters, including number of particles, potential, etc.
- `kwargs...`: FIRE and minimization parameters (all optional, see below).

# Keyword Arguments
- `max_steps::Int=10000`: Maximum number of FIRE steps to perform.
- `tol::Float64=1e-6`: Convergence tolerance for the root mean square force.
- `dt_initial::Float64=0.01`: Initial time step for position updates.
- `dt_max::Float64=0.1`: Maximum allowed time step.
- `alpha0::Float64=0.1`: Initial mixing parameter for velocity update.
- `f_inc::Float64=1.1`: Factor to increase the time step.
- `f_dec::Float64=0.5`: Factor to decrease the time step.
- `Nmin::Int=5`: Number of positive steps before increasing the time step.
- `output_frequency::Int=100`: How often to output trajectory/configuration (if implemented).
- `pathname::String="."`: Directory to write minimized structure (if implemented).
- `traj_name::String="minimized.xyz"`: File name for minimized structure output (if implemented).

# Returns
- `(energy, converged::Bool)`: Final energy and whether minimization converged.

# Notes
- This function modifies `state` in-place.
- The function is designed to be called after running a simulation, starting from the last configuration.
"""
function fire_minimize!(
    state::SimulationState,
    params::Parameters;
    dimension::Int=2,
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
    images = state.images
    system = state.system
    N = params.n_particles
    diameters = state.diameters
    potential = params.potential
    boxl = state.boxl

    # Initialize internal variables
    α = alpha0
    steps_since_neg = 0
    dt = dt_initial
    # The array that holds the information of the velocities
    v = [zeros(StaticArrays.SVector{dimension,Float64}) for _ in 1:N]
    # Use a variable to check convergence
    convergence = false
    # Degrees of freedom
    ndof = dimension * (N - 1.0)

    for step in 1:max_steps
        reset_output!(system.energy_and_forces)
        CellListMap.map_pairwise!(
            (x, y, i, j, d2, output) ->
                energy_and_forces!(x, y, i, j, d2, diameters, output, potential),
            system,
        )
        forces = system.energy_and_forces.forces
        energy = system.energy_and_forces.energy

        F_norm = sqrt(sum(norm(f)^2 for f in forces))

        if step % 100 == 0
            print_fnorm = F_norm / sqrt(ndof)
            print_energy = energy / N
            @info "Step $(step): F_rms = $(print_fnorm), energy = $(print_energy)"
        end

        if F_norm / sqrt(ndof) < tol
            convergence = true
            return energy, convergence
        end

        for i in eachindex(v)
            v[i] = @. v[i] + dt * forces[i]
        end

        P = sum(dot(v[i], forces[i]) for i in eachindex(v))

        v_norm = sqrt(sum(norm(vi)^2 for vi in v))
        f_norm = sqrt(sum(norm(f)^2 for f in forces))
        if v_norm > 0 && f_norm > 0
            scale = α * (v_norm / f_norm)
            for i in eachindex(v)
                v[i] = @. (1.0 - α) * v[i] + scale * forces[i]
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
            fill!(v, zeros(StaticArrays.SVector{dimension,Float64}))
            α = alpha0
            steps_since_neg = 0
        end

        for i in eachindex(system.xpositions)
            x = system.xpositions[i]
            system.xpositions[i] = @. x + dt * v[i]
            system.xpositions[i] = register_images_and_wrap!(
                system.xpositions[i], images[i], boxl
            )
        end
    end

    reset_output!(system.energy_and_forces)
    CellListMap.map_pairwise!(
        (x, y, i, j, d2, output) ->
            energy_and_forces!(x, y, i, j, d2, diameters, output, potential),
        system,
    )
    forces = system.energy_and_forces.EnergyAndForces.forces
    F_norm = sqrt(sum(norm(f)^2 for f in forces))
    F_norm /= sqrt(ndof)

    @warn "FIRE did not converge after $(max_steps) steps; final F_norm = $(F_norm)"

    return nothing
end
