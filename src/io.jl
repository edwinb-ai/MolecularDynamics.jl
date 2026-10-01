"""
    save_log_times_to_file(logs, logn, logbase, filename)

Write the logarithmic snapshot steps to `filename`, one per line, after a metadata header.
"""
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

"""
    generate_log_times(; max_iter=10000, logn=40, logbase=1.35) -> Vector{Int}

Sorted, unique snapshot steps spaced as `logbase^i` within cycles of `logbase^logn` steps.
Also saves them to `new-log-times.txt`.
"""
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

Write a configuration in extended XYZ format: a 3×3 `Lattice` (box vectors one after
another), `pbc`, and per particle its type, id, radius and position. 2D systems are
embedded in 3D with `z = 0`, a unit box vector along z and `pbc="T T F"`.
"""
function write_to_file(
    filepath, step, unitcell, n_particles, positions, diameters, dimension; mode="a"
)
    # Box vectors as the columns of a 3x3 matrix, padded with a unit z vector in 2D
    boxmat = Matrix{Float64}(I, 3, 3)
    boxmat[1:dimension, 1:dimension] .= unitcell
    pbc = dimension == 3 ? "T T T" : "T T F"

    open(filepath, mode) do io
        println(io, n_particles)
        # Column-major order lists the box vectors one after another
        flat_lattice = join(string.(vec(boxmat)), " ")
        Printf.@printf(
            io,
            "Lattice=\"%s\" Properties=type:I:1:id:I:1:radius:R:1:pos:R:3 pbc=\"%s\" Time=%.6g\n",
            flat_lattice,
            pbc,
            step,
        )

        # Write particles (for both 2D and 3D)
        for i in 1:n_particles
            pos = positions[i]
            Printf.@printf(io, "%d %d %lf", 1, i, diameters[i] / 2.0)
            for d in 1:3
                Printf.@printf(io, " %lf", d <= dimension ? pos[d] : 0.0)
            end
            Printf.@printf(io, "\n")
        end
    end
    return nothing
end

"""
    unwrapped(p, img, boxmat)

Unwrapped position of a particle from its wrapped position `p`, image counter `img` and
the 3×3 box matrix `boxmat`. 2D inputs are embedded in 3D.
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

Write a LAMMPS trajectory frame for any box: orthogonal, triclinic, or general triclinic
(see [`write_lammps_box`](@ref)). `unitcell` is a square matrix whose columns are the box
vectors, and the box origin is at zero.
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
        write_lammps_box(io, boxmat, dimension)

        if dimension == 2
            Printf.@printf(io, "ITEM: ATOMS id type radius x y xu yu\n")
        elseif dimension == 3
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
    write_lammps_box(io, boxmat, dimension)

Write the `ITEM: BOX BOUNDS` section of a LAMMPS dump for the 3×3 box `boxmat` (columns are
the box vectors). An upper-triangular box uses the orthogonal or restricted triclinic
format (`xy xz yz` tilts with LAMMPS bounding-box bounds); any other box uses the general
triclinic format (`abc origin`). 2D boxes span z from -0.5 to 0.5, as in LAMMPS.
"""
function write_lammps_box(io, boxmat, dimension)
    M = copy(boxmat)
    zlo = 0.0
    if dimension == 2
        M[3, 3] = 1.0
        zlo = -0.5
    end

    restricted = iszero(M[2, 1]) && iszero(M[3, 1]) && iszero(M[3, 2]) && all(>(0), diag(M))
    if !restricted
        Printf.@printf(io, "ITEM: BOX BOUNDS abc origin pp pp pp\n")
        origin = (0.0, 0.0, zlo)
        for k in 1:3
            Printf.@printf(
                io, "%.16g %.16g %.16g %.16g\n", M[1, k], M[2, k], M[3, k], origin[k]
            )
        end
    elseif isdiag(M)
        Printf.@printf(io, "ITEM: BOX BOUNDS pp pp pp\n")
        Printf.@printf(io, "%.16g %.16g\n", 0.0, M[1, 1])
        Printf.@printf(io, "%.16g %.16g\n", 0.0, M[2, 2])
        Printf.@printf(io, "%.16g %.16g\n", zlo, zlo + M[3, 3])
    else
        (xy, xz, yz) = (M[1, 2], M[1, 3], M[2, 3])
        # LAMMPS writes the bounds of the box's bounding box, then the tilt factors
        xlo_bound = min(0.0, xy, xz, xy + xz)
        xhi_bound = M[1, 1] + max(0.0, xy, xz, xy + xz)
        ylo_bound = min(0.0, yz)
        yhi_bound = M[2, 2] + max(0.0, yz)
        Printf.@printf(io, "ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
        Printf.@printf(io, "%.16g %.16g %.16g\n", xlo_bound, xhi_bound, xy)
        Printf.@printf(io, "%.16g %.16g %.16g\n", ylo_bound, yhi_bound, xz)
        Printf.@printf(io, "%.16g %.16g %.16g\n", zlo, zlo + M[3, 3], yz)
    end

    return nothing
end

"""
    read_file(filepath; dimension=3) -> (unitcell, positions, diameters)

Read a configuration written by [`write_to_file`](@ref). Also reads files from versions
before 0.8.1, which stored 2D systems with a 2×2 `Lattice` and two coordinates.
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
        if m === nothing
            error("Could not parse Lattice property in file header")
        end
        box_entries = parse.(Float64, split(m.captures[1]))
        if length(box_entries) == 9
            unitcell .= reshape(box_entries, 3, 3)[1:dimension, 1:dimension]
        elseif length(box_entries) == dimension^2
            unitcell .= reshape(box_entries, dimension, dimension)
        else
            error("Lattice has $(length(box_entries)) entries, expected 9")
        end

        for _ in 1:n_particles
            line = split(readline(io))
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

"""
    compress_zstd(filepath)

Compress `filepath` to `filepath.zst` and remove the original.
"""
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

"""
    open_files(pathname, traj_name, thermo_name) -> (trajectory_file, thermo_file)

Paths of the output files inside `pathname`, removing previous versions of them.
"""
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
