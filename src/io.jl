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

"""
    write_to_file_lammps(
        filepath, step, unitcell, n_particles, positions, images, diameters, dimension; mode="w"
    )

Write a LAMMPS trajectory file supporting generic simulation boxes (orthogonal or triclinic).
- `unitcell` can be a scalar, vector, or matrix describing the box.
"""
function write_to_file_lammps(
    filepath, step, unitcell, n_particles, positions, images, diameters, dimension; mode="w"
)
    open(filepath, mode) do io
        Printf.@printf(io, "ITEM: TIMESTEP\n%d\n", step)
        Printf.@printf(io, "ITEM: NUMBER OF ATOMS\n%d\n", n_particles)

        # Convert unitcell to a 3x3 matrix
        boxmat = if isa(unitcell, Number)
            unitcell * I(3)
        elseif isa(unitcell, AbstractVector)
            Diagonal(append!(collect(unitcell), fill(1.0, 3 - length(unitcell))))
        elseif isa(unitcell, AbstractMatrix)
            m = zeros(3, 3)
            m[1:size(unitcell, 1), 1:size(unitcell, 2)] .= unitcell
            m
        else
            error("Unsupported unitcell type: $(typeof(unitcell))")
        end

        # Extract box bounds and tilt factors for LAMMPS (see LAMMPS documentation)
        if dimension == 2
            lx = norm(boxmat[:, 1])
            ly = norm(boxmat[:, 2])
            xlo, xhi = 0.0, lx
            ylo, yhi = 0.0, ly
            zlo, zhi = 0.0, 1.0
            xy = boxmat[1, 2]
            # 2D: Only use x/y/xy bounds, z is dummy
            Printf.@printf(io, "ITEM: BOX BOUNDS xy pp pp\n")
            Printf.@printf(io, "%lf %lf %lf\n", xlo, xhi, xy)
            Printf.@printf(io, "%lf %lf 0.0\n", ylo, yhi)
            Printf.@printf(io, "%lf %lf 0.0\n", zlo, zhi)
            Printf.@printf(io, "ITEM: ATOMS id type radius x y xu yu\n")
        elseif dimension == 3
            # LAMMPS expects box bounds and tilt factors: xlo xhi, ylo yhi, zlo zhi, xy, xz, yz
            xlo, xhi = 0.0, boxmat[1, 1]
            ylo, yhi = 0.0, boxmat[2, 2]
            zlo, zhi = 0.0, boxmat[3, 3]
            xy = boxmat[1, 2]
            xz = boxmat[1, 3]
            yz = boxmat[2, 3]
            Printf.@printf(io, "ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
            Printf.@printf(io, "%lf %lf %lf\n", xlo, xhi, xy)
            Printf.@printf(io, "%lf %lf %lf\n", ylo, yhi, yz)
            Printf.@printf(io, "%lf %lf %lf\n", zlo, zhi, xz)
            Printf.@printf(io, "ITEM: ATOMS id type radius x y z xu yu zu\n")
        else
            error("Unsupported dimension: $dimension")
        end

        # Helper to unwrap position (x + n * boxmat)
        function unwrapped(p, img)
            # for generic box, unwrapped = p + boxmat * img
            return p + boxmat * img
        end

        for i in eachindex(diameters, positions, images)
            particle = positions[i]
            image = images[i]
            uw = unwrapped(particle, image)
            if dimension == 2
                Printf.@printf(
                    io,
                    "%d %d %lf %lf %lf %lf %lf\n",
                    i,
                    1,
                    diameters[i] / 2.0,
                    particle[1],
                    particle[2],
                    uw[1],
                    uw[2]
                )
            elseif dimension == 3
                Printf.@printf(
                    io,
                    "%d %d %lf %lf %lf %lf %lf %lf %lf\n",
                    i,
                    1,
                    diameters[i] / 2.0,
                    particle[1],
                    particle[2],
                    particle[3],
                    uw[1],
                    uw[2],
                    uw[3]
                )
            end
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
