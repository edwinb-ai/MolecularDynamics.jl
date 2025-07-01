function finalize_simulation!(
    trajectory_file::String,
    pathname::String,
    total_steps::Int,
    state::SimulationState,
    params::Parameters,
    compress::Bool=false,
)
    final_configuration = joinpath(pathname, "final.xyz")
    write_to_file(
        final_configuration,
        total_steps,
        state.boxl,
        params.n_particles,
        state.system.positions,
        state.diameters,
        state.dimension;
        mode="w",
    )

    if compress && isfile(trajectory_file)
        compress_zstd(trajectory_file)
    end

    return nothing
end

""" This implementation can deal with NVT and NVE by correctly sampling
either one of the ensembles"""
function run_simulation!(
    state::SimulationState,
    params::Parameters,
    ensemble::Ensemble,
    total_steps::Int,
    frequency::Int,
    pathname::String;
    traj_name::String="trajectory.xyz",
    thermo_name::String="thermo.txt",
    compress::Bool=false,
    log_times::Bool=false,
)
    # Remove the files if they existed, and return the files handles
    (trajectory_file, thermo_file) = open_files(pathname, traj_name, thermo_name)
    format_string = Printf.Format("%d %.6f %.6f %.6f\n")
    # Write the columns for the thermo file
    open(thermo_file, "a") do io
        println(io, "# Step Energy Temperature Pressure")
    end

    # Extract parameters from the state
    system = state.system
    velocities = state.velocities
    diameters = state.diameters
    images = state.images
    dimension = state.dimension
    potential = params.potential

    # Compute the volume
    volume = state.boxl^dimension

    # Variables to accumulate results
    virial = 0.0
    nprom = 0
    kinetic_temperature = 0.0

    # We check whether we want logarithmic scale, create variables that can be seen from outside the scope only if necessary
    if log_times
        local snapshot_times = generate_log_times()
        insert!(snapshot_times, 1, 0)
        local current_snapshot_index = 1
    end

    for step in 0:(total_steps - 1)
        # Perform integration
        integrate_half!(
            system.positions,
            images,
            velocities,
            system.energy_and_forces.forces,
            params.dt,
            state.boxl,
        )
        reset_output!(system.energy_and_forces)
        CellListMap.map_pairwise!(
            (x, y, i, j, d2, output) ->
                energy_and_forces!(x, y, i, j, d2, diameters, output, potential),
            system,
        )
        integrate_second_half!(velocities, system.energy_and_forces.forces, params.dt)

        # Apply ensemble-specific logic
        temperature = ensemble_step!(ensemble, velocities, params, state, step + 1)

        # Accumulate values for thermodynamics
        if mod(step, 10) == 0
            virial += system.energy_and_forces.virial
            kinetic_temperature += temperature
            nprom += 1
        end

        # Output thermodynamic quantities periodically
        if mod(step, frequency) == 0
            ener_part = system.energy_and_forces.energy / params.n_particles
            avg_temp = kinetic_temperature / nprom
            pressure = virial / (dimension * nprom * volume) + params.ρ * avg_temp
            open(thermo_file, "a") do io
                Printf.format(io, format_string, step, ener_part, avg_temp, pressure)
            end
            virial, kinetic_temperature, nprom = 0.0, 0.0, 0
        end

        # Output trajectory periodically
        if mod(step, frequency) == 0
            write_to_file_lammps(
                trajectory_file,
                step,
                state.boxl,
                params.n_particles,
                system.positions,
                images,
                diameters,
                dimension;
                mode="a",
            )
        end

        if log_times
            snap_step = snapshot_times[current_snapshot_index]
            if snap_step == step
                # Write to file
                filename = joinpath(pathname, "snapshot.$(snap_step)")
                write_to_file_lammps(
                    filename,
                    snap_step,
                    state.boxl,
                    params.n_particles,
                    system.positions,
                    images,
                    diameters,
                    dimension;
                    mode="w",
                )
                current_snapshot_index += 1
            end
        end
    end

    # Final output and cleanup
    finalize_simulation!(trajectory_file, pathname, total_steps, state, params, compress)

    return nothing
end

""" This implementation is intended to implement Brownian dynamics"""
function run_simulation!(
    state::SimulationState,
    params::Parameters,
    ensemble::Brownian,
    total_steps::Int,
    frequency::Int,
    pathname::String;
    traj_name::String="trajectory.xyz",
    thermo_name::String="thermo.txt",
    compress::Bool=false,
    log_times::Bool=false,
)
    # Remove the files if they existed, and return the files handles
    (trajectory_file, thermo_file) = open_files(pathname, traj_name, thermo_name)
    format_string = Printf.Format("%d %.6f %.6f %.6f\n")
    # Write the columns for the thermo file
    open(thermo_file, "a") do io
        println(io, "# Step Energy Temperature Pressure")
    end

    # Extract parameters from the state
    system = state.system
    diameters = state.diameters
    images = state.images
    dimension = state.dimension
    ktemp = ensemble.ktemp
    potential = params.potential

    # Compute the volume
    volume = state.boxl^dimension
    # Compute the noise term of the diffusion
    sigma = sqrt(2.0 * params.dt)

    # Variables to accumulate results
    virial = 0.0
    nprom = 0
    kinetic_temperature = 0.0

    # We check whether we want logarithmic scale, create variables that can be seen from outside the scope only if necessary
    if log_times
        local snapshot_times = generate_log_times()
        insert!(snapshot_times, 1, 0)
        local current_snapshot_index = 1
    end

    for step in 0:(total_steps - 1)
        reset_output!(system.energy_and_forces)
        CellListMap.map_pairwise!(
            (x, y, i, j, d2, output) ->
                energy_and_forces!(x, y, i, j, d2, diameters, output, potential),
            system,
        )
        # Perform integration
        integrate_brownian!(
            system.positions,
            images,
            system.energy_and_forces.forces,
            params.dt,
            state.boxl,
            state.rng,
            state.dimension,
            ktemp,
            sigma,
        )

        # Accumulate values for thermodynamics
        if mod(step, 10) == 0
            virial += system.energy_and_forces.virial
            nprom += 1
        end

        # Output thermodynamic quantities periodically
        if mod(step, frequency) == 0
            ener_part = system.energy_and_forces.energy / params.n_particles
            pressure = virial / (dimension * nprom * volume) + params.ρ * ktemp
            open(thermo_file, "a") do io
                Printf.format(io, format_string, step, ener_part, ktemp, pressure)
            end
            virial, kinetic_temperature, nprom = 0.0, 0.0, 0
        end

        # Output trajectory periodically
        if mod(step, frequency) == 0
            write_to_file_lammps(
                trajectory_file,
                step,
                state.boxl,
                params.n_particles,
                system.positions,
                images,
                diameters,
                dimension;
                mode="a",
            )
        end

        if log_times
            snap_step = snapshot_times[current_snapshot_index]
            if snap_step == step
                # Write to file
                filename = joinpath(pathname, "snapshot.$(snap_step)")
                write_to_file_lammps(
                    filename,
                    snap_step,
                    state.boxl,
                    params.n_particles,
                    system.positions,
                    images,
                    diameters,
                    dimension;
                    mode="w",
                )
                current_snapshot_index += 1
            end
        end
    end

    # Final output and cleanup
    finalize_simulation!(trajectory_file, pathname, total_steps, state, params, compress)

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
    boxl = state.boxl
    positions = state.system.xpositions
    diameters = state.diameters
    n_particles = params.n_particles
    write_to_file(
        joinpath(pathname, save_config),
        0,
        boxl,
        n_particles,
        positions,
        diameters,
        dimension,
    )

    return nothing
end
