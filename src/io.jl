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
    write_to_file(filepath, step, unitcell, n_particles, positions, diameters, dimension; mode="a")
Writes the system state to file, expecting a matrix for the box/unitcell.
"""
function write_to_file(
    filepath, step, unitcell, n_particles, positions, diameters, dimension; mode="a"
)
    open(filepath, mode) do io
        println(io, n_particles)
        # Write matrix as Lattice property (flattened row-major)
        flat_lattice = join(
            [string(unitcell[i, j]) for i in 1:dimension, j in 1:dimension], " "
        )
        Printf.@printf(
            io,
            "Lattice=\"%s\" Properties=type:I:1:id:I:1:radius:R:1:pos:R:%d Time=%.6g\n",
            flat_lattice,
            dimension,
            step,
        )

        # Write particles (for both 2D and 3D)
        for i in 1:n_particles
            pos = positions[i]
            Printf.@printf(io, "%d %d %lf", 1, i, diameters[i] / 2.0)
            for d in 1:dimension
                Printf.@printf(io, " %lf", pos[d])
            end
            Printf.@printf(io, "\n")
        end
    end
    return nothing
end

"""
    unwrapped(pos, image, unitcell)

Compute the unwrapped (absolute) position of a particle in periodic boundary conditions,
given its position `pos`, image vector `image`, and the simulation box matrix `boxmat`.
"""
@inline function unwrapped(p, img, boxmat)
    if length(p) == 2
        p3 = SVector{3,Float64}(p[1], p[2], 0.0)
        img3 = SVector{3,Int}(img[1], img[2], 0)
        return p3 + boxmat * img3
    else
        return p + boxmat * img
    end
end

"""
    write_to_file_lammps(
        filepath, step, unitcell, n_particles, positions, images, diameters, dimension; mode="w"
    )

Write a LAMMPS trajectory file supporting generic simulation boxes (orthogonal or triclinic).
- `unitcell` should be a square matrix (SMatrix or Matrix).
"""
function write_to_file_lammps(
    filepath, step, unitcell, n_particles, positions, images, diameters, dimension; mode="w"
)
    open(filepath, mode) do io
        Printf.@printf(io, "ITEM: TIMESTEP\n%d\n", step)
        Printf.@printf(io, "ITEM: NUMBER OF ATOMS\n%d\n", n_particles)

        # Use a 3x3 matrix for box representation (for LAMMPS, pad with identity if 2D)
        boxmat = zeros(3, 3)
        boxmat[1:dimension, 1:dimension] .= unitcell

        if dimension == 2
            lx = norm(boxmat[:, 1])
            ly = norm(boxmat[:, 2])
            xlo, xhi = 0.0, lx
            ylo, yhi = 0.0, ly
            zlo, zhi = 0.0, 1.0
            xy = boxmat[1, 2]
            Printf.@printf(io, "ITEM: BOX BOUNDS xy pp pp\n")
            Printf.@printf(io, "%lf %lf %lf\n", xlo, xhi, xy)
            Printf.@printf(io, "%lf %lf 0.0\n", ylo, yhi)
            Printf.@printf(io, "%lf %lf 0.0\n", zlo, zhi)
            Printf.@printf(io, "ITEM: ATOMS id type radius x y xu yu\n")
        elseif dimension == 3
            # Extract box bounds and tilt factors for LAMMPS
            xlo, xhi = 0.0, norm(boxmat[:, 1])
            ylo, yhi = 0.0, norm(boxmat[:, 2])
            zlo, zhi = 0.0, norm(boxmat[:, 3])
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

        for i in eachindex(diameters, positions, images)
            particle = positions[i]
            image = images[i]
            uw = unwrapped(particle, image, boxmat)
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
Reads a configuration file, expecting a matrix for the box/unitcell.
"""
function read_file(filepath; dimension=3)
    n_particles = 0
    unitcell = zeros(dimension, dimension)
    positions = StaticArrays.SVector{dimension,Float64}[]
    radii = Float64[]

    open(filepath, "r") do io
        n_particles = parse(Int64, readline(io))
        header = readline(io)
        m = match(r"Lattice=\"([^\"]+)\"", header)
        if m !== nothing
            box_entries = parse.(Float64, split(m.captures[1]))
            unitcell .= reshape(box_entries, dimension, dimension)
        else
            error("Could not parse Lattice property in file header")
        end

        for _ in 1:n_particles
            line = split(readline(io), " ")
            # type, id, radius, x, y, (z)
            radius = parse(Float64, line[3])
            coords = parse.(Float64, line[4:(3 + dimension)])
            push!(radii, radius)
            push!(positions, StaticArrays.SVector{dimension,Float64}(coords))
        end
    end

    diameters = radii .* 2.0
    return unitcell, positions, diameters
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
