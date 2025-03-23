function save_log_times_to_file(
    logs::Vector{Int}, logn::Int, logbase::Float64, filename::String
)
    open(filename, "w") do file
        # Write metadata as a comment
        write(file, "#maxsnap=$logn,base=$logbase\n")

        # Write each log time
        for log in logs
            write(file, "$log\n")
        end
    end

    return nothing
end

function generate_log_times(; max_iter::Int=10000, logn::Int=40, logbase::Float64=1.35)
    dtime = Int[]
    maxlog = floor(Int, logbase^logn)

    for j in 0:max_iter
        for i in 0:logn
            dt = floor(Int, j * maxlog + logbase^i)
            push!(dtime, dt)
        end
    end

    # Remove duplicates and sort the list
    logs = sort(unique(dtime))

    # Save log to file
    save_log_times_to_file(logs, logn, logbase, "new-log-times.txt")

    # Return the results
    return logs
end

function write_to_file(
    filepath, step, boxl, n_particles, positions, diameters, dimension; mode="a"
)
    open(filepath, mode) do io
        # Write header information
        println(io, n_particles)
        Printf.@printf(
            io,
            "Lattice=\"%lf 0.0 0.0 0.0 %lf 0.0 0.0 0.0 %lf\" Properties=type:I:1:id:I:1:radius:R:1:pos:R:%d Time=%.6g\n",
            boxl,
            boxl,
            boxl,
            dimension,
            step,
        )

        # Select a specialized printing function based on dimension
        write_particle = if dimension == 2
            (io, i, diameters, particle) -> Printf.@printf(
                io,
                "%d %d %lf %lf %lf\n",
                1,
                i,
                diameters[i] / 2.0,
                particle[1],
                particle[2]
            )
        elseif dimension == 3
            (io, i, diameters, particle) -> Printf.@printf(
                io,
                "%d %d %lf %lf %lf %lf\n",
                1,
                i,
                diameters[i] / 2.0,
                particle[1],
                particle[2],
                particle[3]
            )
        else
            error("Unsupported dimension: $dimension")
        end

        # Use the specialized function for each particle
        for i in eachindex(diameters, positions)
            particle = positions[i]
            write_particle(io, i, diameters, particle)
        end
    end

    return nothing
end

function write_to_file_lammps(
    filepath, step, boxl, n_particles, positions, images, diameters, dimension; mode="w"
)
    open(filepath, mode) do io
        # Write header information
        Printf.@printf(io, "ITEM: TIMESTEP\n%d\n", step)
        Printf.@printf(io, "ITEM: NUMBER OF ATOMS\n%d\n", n_particles)

        # Set header and define a closure for per-particle printing
        atom_print = if dimension == 2
            # For 2D, only two box boundaries and atoms coordinates are used.
            Printf.@printf(io, "ITEM: BOX BOUNDS pp pp\n0.0 %lf\n0.0 %lf\n0.0 0.1\n", boxl, boxl)
            Printf.@printf(io, "ITEM: ATOMS id type radius x y xu yu\n")
            (i, diameters, particle, image) -> begin
                # Compute the unwrapped coordinates (2D)
                unwrapped = particle .+ image .* boxl
                Printf.@printf(
                    io,
                    "%d %d %lf %lf %lf %lf %lf\n",
                    i,
                    1,
                    diameters[i] / 2.0,
                    particle[1],
                    particle[2],
                    unwrapped[1],
                    unwrapped[2]
                )
            end
        elseif dimension == 3
            # For 3D, all three dimensions are written.
            Printf.@printf(
                io,
                "ITEM: BOX BOUNDS pp pp pp\n0.0 %lf\n0.0 %lf\n0.0 %lf\n",
                boxl,
                boxl,
                boxl
            )
            Printf.@printf(io, "ITEM: ATOMS id type radius x y z xu yu zu\n")
            (i, diameters, particle, image) -> begin
                # Compute the unwrapped coordinates (3D)
                unwrapped = particle .+ image .* boxl
                Printf.@printf(
                    io,
                    "%d %d %lf %lf %lf %lf %lf %lf %lf\n",
                    i,
                    1,
                    diameters[i] / 2.0,
                    particle[1],
                    particle[2],
                    particle[3],
                    unwrapped[1],
                    unwrapped[2],
                    unwrapped[3]
                )
            end
        else
            error("Unsupported dimension: $dimension")
        end

        # Loop over particles and use the specialized printing closure.
        for i in eachindex(diameters, positions, images)
            particle = positions[i]
            image = images[i]
            atom_print(i, diameters, particle, image)
        end
    end

    return nothing
end

function read_file(filepath)
    # Initialize the variables with a default value
    n_particles = 0
    box_l = 0.0
    positions = []
    radii = []

    open(filepath, "r") do io
        # First line is the number of particles
        line = readline(io)
        n_particles = parse(Int64, line)

        # The second line we can skip for now
        line = split(readline(io), " ")
        # The fifth position included the box size length information
        box_l = parse(Float64, line[5])

        # Now read each line and gather the information
        positions = [@SVector(zeros(2)) for _ in 1:n_particles]
        radii = zeros(n_particles)

        for i in 1:n_particles
            line = split(readline(io), " ")
            parsed_line = parse.(Float64, line)
            positions[i] = SVector{2}(parsed_line[4:end])
            radii[i] = parsed_line[3]
        end
    end

    # The information read are the radii, so convert to diameters
    diameters = radii .* 2.0

    return box_l, positions, diameters
end

function compress_zstd(filepath)
    # Attach the suffix to the original file
    output_file = filepath * ".zst"

    open(filepath, "r") do infile
        # Open the output file for writing, with gzip compression
        open(ZstdCompressorStream, output_file, "w") do outfile
            # Write the contents of the input file to the compressed output file
            write(outfile, read(infile))
        end
    end

    # To avoid having double the files, we delete the original one
    rm(filepath)

    return nothing
end

function open_files(pathname, traj_name, thermo_name)
    # Open files for trajectory and other things
    trajectory_file = joinpath(pathname, traj_name)
    thermo_file = joinpath(pathname, thermo_name)

    files = [trajectory_file, thermo_file]

    for file in files
        if isfile(file)
            rm(file)
        end
    end

    return (trajectory_file, thermo_file)
end
