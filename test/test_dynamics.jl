@testset "Velocities" begin
    rng = Xoshiro(10)
    for dimension in (2, 3)
        v = initialize_velocities(1.7, rng, 500, dimension)
        @test eltype(v) == SVector{dimension,Float64}
        @test norm(sum(v)) < 1e-10
        @test MD.compute_temperature(v, dimension * 499.0) ≈ 1.7
    end
    @test initial_temperature_for_velocities(1.3) == 1.3
    @test initial_temperature_for_velocities(LinearRamp(2.0, 0.5, 10)) == 2.0
end

@testset "Temperature ramps" begin
    ramp = LinearRamp(2.0, 1.0, 11)
    @test ramp(1) == 2.0
    @test ramp(6) ≈ 1.5
    @test ramp(11) == 1.0
    @test ramp(500) == 1.0
    @test ramp(0) == 2.0
    ramp = ExponentialRamp(1.0, 0.1, 101)
    @test ramp(1) ≈ 1.0
    @test ramp(51) ≈ sqrt(0.1)
    @test ramp(101) ≈ 0.1
    @test ramp(1000) == 0.1
    @test all(diff([ramp(k) for k in 1:101]) .< 0)
end

@testset "NVE energy and momentum conservation" begin
    # Smooth XPLOR cutoff so the energy is conserved up to O(dt²) fluctuations
    pot = LennardJonesXPLOR(1.0, 1.0, 2.2, 2.5, false)
    (state, params, L) = lj_state(4; rho=0.8, jitter=0.1, potential=pot)
    params = Parameters(0.8, params.n_particles, 0.002, pot)
    state.velocities = initialize_velocities(1.0, state.rng, params.n_particles, 3)
    dir = mktempdir()
    run_simulation!(state, params, NVE(), 4000, 50, dir)
    thermo = read_thermo(joinpath(dir, "thermo.txt"))
    N = params.n_particles
    total = thermo[:, 2] .+ thermo[:, 3] .* (3 * (N - 1) / (2N))
    @test maximum(abs.(total .- total[1])) < 2e-4 * abs(total[1])
    @test norm(sum(state.velocities)) < 1e-10
    # The lattice melted at this temperature and the run rebuilt the list several times
    @test thermo[end, 2] > thermo[1, 2] + 0.5
    @test state.neighbors.nbuilds > 10
end

@testset "NVT reaches the target temperature" begin
    (state, params, L) = lj_state(4; rho=0.8442)
    state.velocities = initialize_velocities(0.5, state.rng, params.n_particles, 3)
    dir = mktempdir()
    run_simulation!(state, params, NVT(1.5, 0.1), 6000, 20, dir)
    thermo = read_thermo(joinpath(dir, "thermo.txt"))
    temperatures = thermo[(end ÷ 2):end, 3]
    @test sum(temperatures) / length(temperatures) ≈ 1.5 rtol = 0.03
    @test all(isfinite, thermo)
    @test isfile(joinpath(dir, "trajectory.xyz"))
    @test isfile(joinpath(dir, "final.xyz"))
    # Positions are left inside the box after a run
    @test all(all(0 .<= inv(state.unitcell) * x .< 1) for x in state.positions)
end

@testset "NVT with a temperature ramp" begin
    (state, params, L) = lj_state(3; rho=0.8442)
    state.velocities = initialize_velocities(2.0, state.rng, params.n_particles, 3)
    dir = mktempdir()
    ramp = LinearRamp(2.0, 0.5, 2000)
    run_simulation!(state, params, NVT(ramp, 0.05), 3000, 10, dir)
    thermo = read_thermo(joinpath(dir, "thermo.txt"))
    @test sum(thermo[(end - 50):end, 3]) / 51 ≈ 0.5 rtol = 0.1
end

@testset "Runs are independent of the number of tasks" begin
    function trajectory(nchunks, ntasks)
        (state, params, L) = lj_state(4; jitter=0.1, seed=3)
        state.neighbors = MD.NeighborList(
            state.positions, state.unitcell, 2.5; nchunks, ntasks
        )
        state.velocities = initialize_velocities(2.0, Xoshiro(1), params.n_particles, 3)
        run_simulation!(state, params, NVE(), 200, 1000, mktempdir())
        return [x + state.unitcell * img for (x, img) in zip(state.positions, state.images)]
    end
    reference = trajectory(1, 1)
    @test maximum(norm.(trajectory(16, 4) .- reference)) < 1e-8
end

@testset "2D molecular dynamics" begin
    rng = Xoshiro(11)
    (n, ρ) = (20, 0.7)
    L = n / sqrt(ρ)
    positions = [SVector(i + 0.5, j + 0.5) * (L / n) for i in 0:(n - 1) for j in 0:(n - 1)]
    params = Parameters(ρ, length(positions), 0.005, LennardJones(; r_cut=2.5))
    state = quiet() do
        initialize_state(
            params,
            mktempdir();
            dimension=2,
            cutoff=2.5,
            rng=rng,
            unitcell=[L, L],
            positions=positions,
            diameters=ones(length(positions)),
        )
    end
    state.velocities = initialize_velocities(1.0, rng, params.n_particles, 2)
    dir = mktempdir()
    run_simulation!(state, params, NVT(1.0, 0.1), 2000, 50, dir)
    thermo = read_thermo(joinpath(dir, "thermo.txt"))
    @test all(isfinite, thermo)
    @test sum(thermo[(end ÷ 2):end, 3]) / length(thermo[(end ÷ 2):end, 3]) ≈ 1.0 rtol = 0.05
end

@testset "Brownian dynamics" begin
    @testset "free diffusion" begin
        # Mean squared displacement of non-interacting particles is 2 d D t with D = 1
        rng = Xoshiro(12)
        (N, L) = (4000, 20.0)
        positions = [L * rand(rng, SVector{3,Float64}) for _ in 1:N]
        params = Parameters(N / L^3, N, 1e-3, Ideal())
        state = quiet() do
            initialize_state(
                params,
                mktempdir();
                cutoff=0.5,
                rng=rng,
                unitcell=[L, L, L],
                positions=positions,
                diameters=ones(N),
            )
        end
        start = [
            x + state.unitcell * img for (x, img) in zip(state.positions, state.images)
        ]
        run_simulation!(state, params, Brownian(1.0), 1000, 500, mktempdir())
        finish = [
            x + state.unitcell * img for (x, img) in zip(state.positions, state.images)
        ]
        msd = sum(norm(finish[i] - start[i])^2 for i in 1:N) / N
        @test msd ≈ 6.0 rtol = 0.05
    end

    @testset "interacting system" begin
        (state, params, L) = lj_state(3; jitter=0.05)
        params = Parameters(params.ρ, params.n_particles, 1e-4, params.potential)
        dir = mktempdir()
        run_simulation!(state, params, Brownian(1.0), 500, 100, dir)
        thermo = read_thermo(joinpath(dir, "thermo.txt"))
        @test size(thermo, 1) == 5
        @test all(isfinite, thermo)
        @test all(thermo[:, 3] .== 1.0)
    end
end

@testset "Velocities must be set" begin
    (state, params, L) = lj_state(2)
    @test_throws ArgumentError run_simulation!(state, params, NVE(), 10, 5, mktempdir())
end
