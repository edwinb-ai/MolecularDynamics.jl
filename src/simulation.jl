function initialize_state(
    params::Parameters,
    ktemp::Float64,
    pathname::String,
    diameters::Vector{Float64};
    from_file::String="",
    dimension::Int=3,
)
    rng = Random.Xoshiro()

    # The degrees of freedom
    nf = dimension * (params.n_particles - 1.0)

    # Initialize the system
    cutoff = 2.5
    inter_distance = (1.0 / params.ρ)^(1 / dimension)
    boxl = (params.n_particles / params.ρ)^(1 / dimension)
    system = init_system(
        boxl,
        cutoff,
        inter_distance,
        rng,
        pathname;
        random=true,
        n_particles=params.n_particles,
    )
    # Initialize the velocities of the system by having the correct temperature
    velocities = initialize_velocities(
        system.positions, ktemp, nf, rng, params.n_particles, dimension
    )
    # Adjust the particles using the velocities
    for i in eachindex(system.positions)
        system.positions[i] = @. system.positions[i] - (velocities[i] * params.dt)
    end
    # Zero out the arrays
    reset_output!(system.energy_and_forces)

    # Initialize the array of images
    images = [
        StaticArrays.@MVector zeros(Int32, Int(dimension)) for _ in eachindex(velocities)
    ]

    state = SimulationState(system, diameters, rng, boxl, velocities, images, dimension, nf)

    return state
end

function ensemble_step!(
    ::NVE, velocities, params::Parameters, nf::Float64, state::SimulationState
)
    return compute_temperature(velocities, nf)
end

function ensemble_step!(
    ensemble::NVT, velocities, params::Parameters, nf::Float64, state::SimulationState
)
    # Apply thermostat, e.g., Bussi thermostat
    bussi!(velocities, ensemble.ktemp, nf, params.dt, ensemble.tau, state.rng)
    return compute_temperature(velocities, nf)
end

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
        state.diameters;
        mode="w",
    )

    if compress && isfile(trajectory_file)
        compress_gz(trajectory_file)
    end

    return nothing
end

function run_simulation!(
    state::SimulationState,
    params::Parameters,
    ensemble::Ensemble,
    total_steps::Int,
    frequency::Int,
    pathname;
    traj_name="trajectory.xyz",
    thermo_name="thermo.txt",
    compress::Bool=false,
    log_times::Bool=false,
)
    # Remove the files if they existed, and return the files handles
    (trajectory_file, thermo_file) = open_files(pathname, traj_name, thermo_name)
    format_string = Printf.Format("%d %.6f %.6f %.6f\n")

    # Extract parameters from the state
    system = state.system
    velocities = state.velocities
    diameters = state.diameters
    images = state.images
    dimension = state.dimension
    nf = state.nf

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
            (x, y, i, j, d2, output) -> energy_and_forces!(x, y, i, j, d2, output), system
        )
        integrate_second_half!(velocities, system.energy_and_forces.forces, params.dt)

        # Apply ensemble-specific logic
        temperature = ensemble_step!(ensemble, velocities, params, nf, state)

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
                diameters;
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
                    diameters;
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