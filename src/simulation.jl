"""
    compute_box_volume(unitcell)

Compute the volume (or area in 2D) of the simulation box, given a square matrix `unitcell`.
This works for any dimension.
"""
@inline function compute_box_volume(unitcell)
    return abs(det(unitcell))
end

"""
    finalize_simulation!(trajectory_file, pathname, total_steps, state, params, compress=false)

Write the final configuration to `pathname/final.xyz` and optionally compress the trajectory.
"""
function finalize_simulation!(
    trajectory_file::String,
    pathname::String,
    total_steps::Int,
    state::SimulationState,
    params::Parameters,
    compress::Bool=false,
)
    final_configuration = joinpath(pathname, "final.xyz")
    (positions, _) = wrapped_positions(state)
    write_to_file(
        final_configuration,
        total_steps,
        state.unitcell,
        params.n_particles,
        positions,
        state.diameters,
        state.dimension;
        mode="w",
    )

    if compress && isfile(trajectory_file)
        compress_zstd(trajectory_file)
    end

    return nothing
end

"""
    run_simulation!(state, params, ensemble, total_steps, frequency, pathname; kwargs...)

Integrate `total_steps` steps with velocity Verlet in the `NVT` or `NVE` `ensemble`,
writing thermodynamics (`thermo_name`) and the trajectory in LAMMPS format (`traj_name`)
every `frequency` steps. Velocities must be set beforehand.

# Keyword arguments
- `traj_name="trajectory.xyz"`, `thermo_name="thermo.txt"`: output file names.
- `compress=false`: compress the trajectory with zstd at the end.
- `log_times=false`: also write snapshots at logarithmically spaced steps.
"""
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
    if length(state.velocities) != length(state.positions)
        throw(
            ArgumentError(
                "velocities are not set, use `state.velocities = initialize_velocities(...)`",
            ),
        )
    end

    # Remove the files if they existed, and return the files handles
    (trajectory_file, thermo_file) = open_files(pathname, traj_name, thermo_name)
    format_string = Printf.Format("%d %.6f %.6f %.6f\n")
    # Write the columns for the thermo file
    open(thermo_file, "a") do io
        return println(io, "# Step Energy Temperature Pressure")
    end

    # Extract parameters from the state
    positions = state.positions
    velocities = state.velocities
    forces = state.forces
    diameters = state.diameters
    images = state.images
    neighbors = state.neighbors
    dimension = state.dimension
    potential = params.potential
    unitcell = state.unitcell

    # Compute the volume
    volume = compute_box_volume(unitcell)

    # We check whether we want logarithmic scale, create variables that can be seen from outside the scope only if necessary
    if log_times
        local snapshot_times = generate_log_times()
        insert!(snapshot_times, 1, 0)
        local current_snapshot_index = 1
    end

    # Velocity Verlet needs the forces of the starting configuration
    build!(neighbors, positions, images)
    compute_forces!(state, potential)

    for step in 0:(total_steps - 1)
        # Perform integration
        integrate_half!(positions, velocities, forces, params.dt)
        update!(neighbors, positions, images)
        compute_forces!(state, potential)
        integrate_second_half!(velocities, forces, params.dt)

        # Apply ensemble-specific logic
        temperature = ensemble_step!(ensemble, velocities, params, state, step + 1)

        # Output thermodynamic quantities and trajectory periodically
        if mod(step, frequency) == 0
            # Always add long-range corrections if needed
            total_energy = state.energy + energy_lrc(potential, params.n_particles, volume)
            # Make the energy per particle
            total_energy /= params.n_particles
            # we compute the pressure with the virial
            pressure = state.virial / (dimension * volume) + params.ρ * temperature
            # Also add the long-range pressure correction
            pressure += pressure_lrc(potential, params.n_particles, volume)
            open(thermo_file, "a") do io
                return Printf.format(
                    io, format_string, step, total_energy, temperature, pressure
                )
            end

            (wrapped, wrapped_images) = wrapped_positions(state)
            write_to_file_lammps(
                trajectory_file,
                step,
                unitcell,
                params.n_particles,
                wrapped,
                wrapped_images,
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
                (wrapped, wrapped_images) = wrapped_positions(state)
                write_to_file_lammps(
                    filename,
                    snap_step,
                    unitcell,
                    params.n_particles,
                    wrapped,
                    wrapped_images,
                    diameters,
                    dimension;
                    mode="w",
                )
                current_snapshot_index += 1
            end
        end
    end

    # Leave the positions wrapped into the box
    build!(neighbors, positions, images)
    # Final output and cleanup
    finalize_simulation!(trajectory_file, pathname, total_steps, state, params, compress)

    return nothing
end

"""
    run_simulation!(state, params, ensemble::Brownian, total_steps, frequency, pathname; kwargs...)

Overdamped Brownian dynamics, see [`integrate_brownian!`](@ref). The pressure written to
the thermo file averages the virial over every 10th step since the previous output.
Keyword arguments are the same as for the molecular dynamics method.
"""
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
        return println(io, "# Step Energy Temperature Pressure")
    end

    # Extract parameters from the state
    positions = state.positions
    forces = state.forces
    diameters = state.diameters
    images = state.images
    neighbors = state.neighbors
    dimension = state.dimension
    ktemp = ensemble.ktemp
    potential = params.potential
    unitcell = state.unitcell

    # Compute the volume
    volume = compute_box_volume(unitcell)
    # Compute the noise term of the diffusion
    sigma = sqrt(2.0 * params.dt)

    # Variables to accumulate results
    virial = 0.0
    nprom = 0

    # We check whether we want logarithmic scale, create variables that can be seen from outside the scope only if necessary
    if log_times
        local snapshot_times = generate_log_times()
        insert!(snapshot_times, 1, 0)
        local current_snapshot_index = 1
    end

    build!(neighbors, positions, images)
    compute_forces!(state, potential)

    for step in 0:(total_steps - 1)
        # Perform integration
        integrate_brownian!(positions, forces, params.dt, state.rng, ktemp, sigma)
        update!(neighbors, positions, images)
        compute_forces!(state, potential)

        # Accumulate values for thermodynamics
        if mod(step, 10) == 0
            virial += state.virial
            nprom += 1
        end

        # Output thermodynamic quantities and trajectory periodically
        if mod(step, frequency) == 0
            if nprom == 0
                virial = state.virial
                nprom = 1
            end
            ener_part = state.energy / params.n_particles
            pressure = virial / (dimension * nprom * volume) + params.ρ * ktemp
            open(thermo_file, "a") do io
                return Printf.format(io, format_string, step, ener_part, ktemp, pressure)
            end
            virial, nprom = 0.0, 0

            (wrapped, wrapped_images) = wrapped_positions(state)
            write_to_file_lammps(
                trajectory_file,
                step,
                unitcell,
                params.n_particles,
                wrapped,
                wrapped_images,
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
                (wrapped, wrapped_images) = wrapped_positions(state)
                write_to_file_lammps(
                    filename,
                    snap_step,
                    unitcell,
                    params.n_particles,
                    wrapped,
                    wrapped_images,
                    diameters,
                    dimension;
                    mode="w",
                )
                current_snapshot_index += 1
            end
        end
    end

    # Leave the positions wrapped into the box
    build!(neighbors, positions, images)
    # Final output and cleanup
    finalize_simulation!(trajectory_file, pathname, total_steps, state, params, compress)

    return nothing
end
