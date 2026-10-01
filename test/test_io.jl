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
