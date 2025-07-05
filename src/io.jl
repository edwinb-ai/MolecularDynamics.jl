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

"""
    write_to_file(
        filepath, step, unitcell, n_particles, positions, diameters, dimension; mode="a"
    )

Write an extended XYZ file with a general unitcell (can be scalar, vector, or matrix).
The Lattice attribute is always 9 numbers, as required for OVITO and similar tools.
For 2D, a 3x3 matrix is constructed: the input fills the upper-left, rest zeros except for unit z-thickness.
"""
function write_to_file(
    filepath, step, unitcell, n_particles, positions, diameters, dimension; mode="a"
)
    open(filepath, mode) do io
        println(io, n_particles)
        # Prepare a 3×3 box matrix in column-major order, as OVITO expects
        boxmat = zeros(3, 3)
        if isa(unitcell, Number)
            boxmat[1, 1] = unitcell
            boxmat[2, 2] = unitcell
            boxmat[3, 3] = unitcell
        elseif isa(unitcell, AbstractVector)
            for i in eachindex(unitcell)
                boxmat[i, i] = unitcell[i]
            end
            if dimension == 2
                boxmat[3, 3] = 1.0 # artificial thickness for 2D
            end
        elseif isa(unitcell, AbstractMatrix)
            boxmat[1:size(unitcell, 1), 1:size(unitcell, 2)] .= unitcell
            if dimension == 2
                boxmat[3, 3] = 1.0
            end
        else
            error("Unsupported unitcell type: $(typeof(unitcell))")
        end
        # Flatten in column-major order (Julia default)
        lattice = vec(boxmat)
        # Write Lattice header (always 9 numbers)
        Printf.@printf(
            io,
            "Lattice=\"%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\" Properties=type:I:1:id:I:1:radius:R:1:pos:R:%d Time=%.6g\n",
            lattice...,
            dimension,
            step
        )
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
            Printf.@printf(
                io, "ITEM: BOX BOUNDS pp pp\n0.0 %lf\n0.0 %lf\n0.0 0.1\n", boxl, boxl
            )
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

"""
    read_file(filepath; dimension=3)

Reads an extended XYZ file and returns (unitcell, positions, diameters).
Supports general unitcell (matrix), not just cubic box.
"""
function read_file(filepath; dimension=3)
    open(filepath, "r") do io
        n_particles = parse(Int, readline(io))
        header = readline(io)
        # Extract lattice
        lattice_str = match(r"Lattice=\"([^\"]+)\"", header).captures[1]
        lattice_vals = parse.(Float64, split(lattice_str))
        if dimension == 2
            unitcell = [lattice_vals[1] 0.0; 0.0 lattice_vals[4]]
        elseif dimension == 3
            unitcell = reshape(lattice_vals, 3, 3)
        else
            error("Unsupported dimension: $dimension")
        end
        # Read particles
        positions = []
        diameters = []
        for _ in 1:n_particles
            line = readline(io)
            fields = split(line)
            r = parse(Float64, fields[3])
            pos = map(x -> parse(Float64, x), fields[4:end])
            push!(diameters, 2.0 * r)
            push!(positions, pos)
        end
        return unitcell, positions, diameters
    end
end

function compress_zstd(filepath)
    # Attach the suffix to the original file
    output_file = filepath * ".zst"

    open(filepath, "r") do infile
        # Open the output file for writing, with zstd compression
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
