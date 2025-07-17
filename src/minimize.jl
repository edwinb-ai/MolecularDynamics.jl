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
    unitcell = state.unitcell
    # We need the inverse of the box matrix
    unitcell_inv = inv(state.unitcell)

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

    # Define the energy function to avoid allocations
    function pairwise_eval(x, y, i, j, d2, output)
        return energy_and_forces!(x, y, i, j, d2, diameters, output, potential)
    end

    for step in 1:max_steps
        reset_output!(system.energy_and_forces)
        CellListMap.map_pairwise!(pairwise_eval, system)
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
            system.xpositions[i] = wrap_to_box!(
                system.xpositions[i], images[i], unitcell, unitcell_inv
            )
        end
    end

    reset_output!(system.energy_and_forces)
    CellListMap.map_pairwise!(pairwise_eval, system)
    forces = system.energy_and_forces.EnergyAndForces.forces
    F_norm = sqrt(sum(norm(f)^2 for f in forces))
    F_norm /= sqrt(ndof)

    @warn "FIRE did not converge after $(max_steps) steps; final F_norm = $(F_norm)"

    return nothing
end

"""
    minimize!(
        state::SimulationState,
        params::Parameters,
        pathname::String,
        dimension::Int;
        method::Symbol = :FIRE,
        save_config::String = "minimized.xyz",
        kwargs...
    )

Perform energy minimization using the specified `method` (default: `:FIRE`).  
All additional keyword arguments are forwarded to the underlying minimizer.

After minimization, saves the final configuration to a file named `save_config` (default: "minimized.xyz") in the directory given by `pathname`.

# Arguments
- `state::SimulationState`: The current simulation state to be minimized.
- `params::Parameters`: Simulation parameters.
- `pathname::String`: Directory path where the minimized configuration file will be saved.
- `dimension::Int`: Dimensionality of the system (e.g., 2 or 3).
- `method::Symbol`: (Keyword) Minimization method to use. Currently supports `:FIRE` only which implements the standard fast inertial relaxation engine (FIRE) algorithm.
- `save_config::String`: (Keyword) Name of the file to save the minimized configuration.
- `kwargs...`: Additional keyword arguments forwarded to the minimizer.

# Notes
- The configuration is saved using `write_to_file` with the filename constructed from `joinpath(pathname, save_config)`.
- If the specified minimization method is not supported, an error is thrown.
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
        fire_minimize!(state, params; dimension=dimension, kwargs...)
    else
        error("Unknown minimization method: $method")
    end

    # Unpack some variables and save the final configuration to file
    unitcell = state.unitcell
    positions = state.system.xpositions
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

    return nothing
end
