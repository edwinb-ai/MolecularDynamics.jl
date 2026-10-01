@testset "Configuration files" begin
    for (dimension, unitcell) in (
        (3, SMatrix{3,3}([6.0 1.0 0.5; 0.0 5.0 0.7; 0.0 0.0 7.0])),
        (2, SMatrix{2,2}([8.0 2.0; 0.0 6.0])),
    )
        rng = Xoshiro(20)
        positions = [unitcell * rand(rng, SVector{dimension,Float64}) for _ in 1:50]
        diameters = 0.8 .+ 0.4 .* rand(rng, 50)
        path = joinpath(mktempdir(), "config.xyz")
        MD.write_to_file(path, 0, unitcell, 50, positions, diameters, dimension; mode="w")
        (uc, x, d) = MD.read_file(path; dimension=dimension)
        @test uc ≈ unitcell
        @test maximum(norm.(x .- positions)) < 1e-5
        @test maximum(abs.(d .- diameters)) < 1e-5

        # A state can start from that file
        params = Parameters(1.0, 50, 0.005, Gaussian())
        state = quiet() do
            initialize_state(
                params, mktempdir(); dimension=dimension, from_file=path, cutoff=1.5
            )
        end
        @test state.unitcell ≈ unitcell
        @test length(state.positions) == 50
        @test state.diameters ≈ diameters atol = 1e-5
    end
end

@testset "Extended XYZ layout" begin
    function written(unitcell, positions)
        D = size(unitcell, 1)
        path = joinpath(mktempdir(), "config.xyz")
        n = length(positions)
        MD.write_to_file(path, 3, unitcell, n, positions, ones(n), D; mode="w")
        return readlines(path)
    end
    lattice(header) = parse.(Float64, split(match(r"Lattice=\"([^\"]+)\"", header)[1]))

    # 2D is embedded in 3D: 9 lattice entries, three coordinates, no periodicity along z
    lines = written(SMatrix{2,2}([8.0 2.0; 0.0 6.0]), [SVector(1.0, 2.0)])
    @test lattice(lines[2]) == [8.0, 0.0, 0.0, 2.0, 6.0, 0.0, 0.0, 0.0, 1.0]
    @test occursin("pos:R:3", lines[2])
    @test occursin("pbc=\"T T F\"", lines[2])
    @test parse.(Float64, split(lines[3]))[4:6] == [1.0, 2.0, 0.0]

    lines = written(
        SMatrix{3,3}([6.0 1.0 0.5; 0.0 5.0 0.7; 0.0 0.0 7.0]), [SVector(1.0, 2.0, 3.0)]
    )
    @test lattice(lines[2]) == [6.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.5, 0.7, 7.0]
    @test occursin("pbc=\"T T T\"", lines[2])

    # Files written before 0.8.1 stored 2D systems with a 2x2 lattice and two coordinates
    path = joinpath(mktempdir(), "old.xyz")
    write(
        path,
        """
        2
        Lattice="8.0 0.0 2.0 6.0" Properties=type:I:1:id:I:1:radius:R:1:pos:R:2 Time=5
        1 1 0.500000 1.000000 2.000000
        1 2   0.600000  3.500000 4.250000
        """,
    )
    (uc, x, d) = MD.read_file(path; dimension=2)
    @test uc == [8.0 2.0; 0.0 6.0]
    @test x == [SVector(1.0, 2.0), SVector(3.5, 4.25)]
    @test d == [1.0, 1.2]
end

@testset "Wrapped positions and LAMMPS output" begin
    (state, params, L) = lj_state(2; jitter=0.1)
    # Push particles across the box without rebuilding
    state.positions .+= [SVector(0.4 * L, -0.3 * L, 0.0) for _ in state.positions]
    unwrapped = [
        x + state.unitcell * img for (x, img) in zip(state.positions, state.images)
    ]
    before = copy(state.positions)
    (x, images) = wrapped_positions(state)
    @test state.positions == before
    @test all(all(0 .<= inv(state.unitcell) * xi .< 1) for xi in x)
    @test all(x[i] + state.unitcell * images[i] ≈ unwrapped[i] for i in eachindex(x))

    path = joinpath(mktempdir(), "frame.lammpstrj")
    MD.write_to_file_lammps(
        path, 7, state.unitcell, length(x), x, images, state.diameters, 3
    )
    lines = readlines(path)
    @test lines[1] == "ITEM: TIMESTEP"
    @test lines[2] == "7"
    @test parse(Int, lines[4]) == length(x)
    @test startswith(lines[9], "ITEM: ATOMS id type radius x y z xu yu zu")
    @test length(lines) == 9 + length(x)
    columns = parse.(Float64, split(lines[10]))
    @test columns[7:9] ≈ unwrapped[1] atol = 1e-5
end

@testset "Output options" begin
    (state, params, L) = lj_state(2; jitter=0.1)
    state.velocities = initialize_velocities(1.0, state.rng, params.n_particles, 3)
    dir = mktempdir()
    cd(dir) do
        run_simulation!(
            state, params, NVT(1.0, 0.5), 30, 10, dir; compress=true, log_times=true
        )
    end
    @test isfile(joinpath(dir, "trajectory.xyz.zst"))
    @test !isfile(joinpath(dir, "trajectory.xyz"))
    @test isfile(joinpath(dir, "snapshot.0"))
    @test isfile(joinpath(dir, "snapshot.1"))
    @test size(read_thermo(joinpath(dir, "thermo.txt")), 1) == 3
end

@testset "Random initial configuration" begin
    params = Parameters(0.5, 60, 0.005, Gaussian())
    state = quiet() do
        redirect_stdout(devnull) do
            initialize_state(
                params, mktempdir(); dimension=2, random_init=true, rng=Xoshiro(21)
            )
        end
    end
    L = sqrt(60 / 0.5)
    @test state.unitcell ≈ SMatrix{2,2}(L * I(2))
    @test length(state.positions) == 60
    @test all(all(0 .<= inv(state.unitcell) * x .< 1) for x in state.positions)
    # Packmol removes overlaps (tolerance 1.0)
    shortest = minimum(
        norm(
            state.positions[i] - state.positions[j] -
            L * round.((state.positions[i] - state.positions[j]) / L),
        ) for i in 1:59 for j in (i + 1):60
    )
    @test shortest > 0.9
end

@testset "LAMMPS box headers" begin
    function header(unitcell)
        D = size(unitcell, 1)
        path = joinpath(mktempdir(), "frame.lammpstrj")
        x = [unitcell * fill(0.5, SVector{D,Float64})]
        MD.write_to_file_lammps(path, 0, unitcell, 1, x, [zero(SVector{D,Int32})], [1.0], D)
        lines = readlines(path)
        return lines[5], [parse.(Float64, split(l)) for l in lines[6:8]], lines[9]
    end

    # Expected values are those LAMMPS itself writes for the same boxes
    (item, bounds, atoms) = header(SMatrix{3,3}(Diagonal([6.0, 5.0, 7.0])))
    @test item == "ITEM: BOX BOUNDS pp pp pp"
    @test bounds == [[0.0, 6.0], [0.0, 5.0], [0.0, 7.0]]
    @test atoms == "ITEM: ATOMS id type radius x y z xu yu zu"

    (item, bounds, _) = header(SMatrix{3,3}([6.0 1.0 0.5; 0.0 5.0 0.7; 0.0 0.0 7.0]))
    @test item == "ITEM: BOX BOUNDS xy xz yz pp pp pp"
    @test bounds ≈ [[0.0, 7.5, 1.0], [0.0, 5.7, 0.5], [0.0, 7.0, 0.7]]

    (item, bounds, _) = header(SMatrix{3,3}([6.0 -1.0 0.5; 0.0 5.0 -0.7; 0.0 0.0 7.0]))
    @test bounds ≈ [[-1.0, 6.5, -1.0], [-0.7, 5.0, 0.5], [0.0, 7.0, -0.7]]

    (item, bounds, atoms) = header(SMatrix{2,2}([8.0 0.0; 0.0 6.0]))
    @test item == "ITEM: BOX BOUNDS pp pp pp"
    @test bounds == [[0.0, 8.0], [0.0, 6.0], [-0.5, 0.5]]
    @test atoms == "ITEM: ATOMS id type radius x y xu yu"

    (item, bounds, _) = header(SMatrix{2,2}([8.0 2.0; 0.0 6.0]))
    @test item == "ITEM: BOX BOUNDS xy xz yz pp pp pp"
    @test bounds ≈ [[0.0, 10.0, 2.0], [0.0, 6.0, 0.0], [-0.5, 0.5, 0.0]]

    # Not upper triangular: general triclinic, one box vector and origin per line
    unitcell = SMatrix{3,3}([5.0 -1.0 0.5; 2.0 4.0 -1.0; 1.0 1.0 6.0])
    (item, bounds, _) = header(unitcell)
    @test item == "ITEM: BOX BOUNDS abc origin pp pp pp"
    @test bounds ≈ [[unitcell[:, k]; 0.0] for k in 1:3]
end
