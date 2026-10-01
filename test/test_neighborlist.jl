@testset "Neighbor list forces" begin
    @testset "cubic LJ, jittered FCC" begin
        (positions, L) = fcc_positions(5, 0.8442)
        rng = Xoshiro(2)
        positions = [x + 0.3 * (rand(rng, SVector{3,Float64}) .- 0.5) for x in positions]
        unitcell = SMatrix{3,3}(L * I(3))
        d = ones(length(positions))
        pot = LennardJones(; r_cut=2.5)
        (E0, W0, F0) = brute_force(positions, d, unitcell, pot, 2.5)
        # Serial, several chunks on one task, and several tasks sharing the chunks
        for (nchunks, ntasks) in ((1, 1), (3, 1), (8, 3))
            (E, W, F) = package_forces(positions, d, unitcell, pot, 2.5; nchunks, ntasks)
            @test E ≈ E0 rtol = 1e-12
            @test W ≈ W0 rtol = 1e-12
            @test max_force_error(F, F0) < 1e-10 * maximum(norm, F0)
        end
    end

    @testset "polydisperse LJ" begin
        (positions, L) = fcc_positions(4, 0.7)
        rng = Xoshiro(3)
        d = 0.8 .+ 0.4 .* rand(rng, length(positions))
        unitcell = SMatrix{3,3}(L * I(3))
        pot = LennardJones(; r_cut=2.5)
        (E0, W0, F0) = brute_force(positions, d, unitcell, pot, 2.5)
        (E, W, F) = package_forces(positions, d, unitcell, pot, 2.5; nchunks=4, ntasks=2)
        @test E ≈ E0 rtol = 1e-12
        @test W ≈ W0 rtol = 1e-12
        @test max_force_error(F, F0) < 1e-10 * maximum(norm, F0)
    end

    # Random configurations with a bounded potential, in boxes of different shapes and sizes
    boxes = [
        ("3D triclinic", SMatrix{3,3}([7.0 2.1 -1.4; 0.0 6.5 1.6; 0.0 0.0 7.5]), 200),
        ("3D small box (cutoff > L/2)", SMatrix{3,3}(3.0 * I(3)), 20),
        ("3D thin slab", SMatrix{3,3}(Diagonal([8.0, 8.0, 1.5])), 60),
        ("2D square", SMatrix{2,2}(12.0 * I(2)), 150),
        ("2D triclinic", SMatrix{2,2}([10.0 4.0; 0.0 9.0]), 150),
        ("2D small box", SMatrix{2,2}([3.5 1.0; 0.0 3.0]), 15),
    ]
    @testset "$name" for (name, unitcell, N) in boxes
        D = size(unitcell, 1)
        rng = Xoshiro(4)
        positions = [unitcell * rand(rng, SVector{D,Float64}) for _ in 1:N]
        d = ones(N)
        cutoff = 2.5
        K =
            ceil(
                Int, (cutoff + 0.5) / minimum(1 / norm(inv(unitcell)[k, :]) for k in 1:D)
            ) + 1
        (E0, W0, F0) = brute_force(positions, d, unitcell, Gaussian(), cutoff; K=K)
        for (nchunks, ntasks) in ((1, 1), (5, 2))
            (E, W, F) = package_forces(
                positions, d, unitcell, Gaussian(), cutoff; nchunks, ntasks
            )
            @test E ≈ E0 rtol = 1e-12
            @test W ≈ W0 rtol = 1e-12
            @test max_force_error(F, F0) < 1e-12 * max(1.0, maximum(norm, F0))
        end
    end

    @testset "stored pairs match brute-force count" begin
        unitcell = SMatrix{3,3}([7.0 2.1 -1.4; 0.0 6.5 1.6; 0.0 0.0 7.5])
        rng = Xoshiro(5)
        x = [unitcell * rand(rng, SVector{3,Float64}) for _ in 1:150]
        (cutoff, skin) = (2.0, 0.4)
        nl = MD.NeighborList(x, unitcell, cutoff; skin=skin, nchunks=3, ntasks=2)
        MD.build!(nl, x, zeros(SVector{3,Int32}, length(x)))
        shifts = [
            unitcell * SVector{3,Float64}(Tuple(t)) for
            t in CartesianIndices((-2:2, -2:2, -2:2))
        ]
        expected = count(
            norm(x[i] - x[j] - s) < cutoff + skin for i in eachindex(x) for j in
                                                                            i:length(x) for
            s in shifts if !(i == j && iszero(s))
        )
        # Self-image pairs appear once for each ±shift pair, so this count is exact
        @test MD.npairs(nl) ==
            expected -
              count(norm(s) < cutoff + skin for s in shifts if !iszero(s)) * length(x) ÷ 2
    end
end

@testset "Neighbor list updates" begin
    (positions, L) = fcc_positions(4, 0.8442)
    unitcell = SMatrix{3,3}(L * I(3))
    d = ones(length(positions))
    pot = LennardJones(; r_cut=2.5)
    skin = 0.4
    x = copy(positions)
    images = zeros(SVector{3,Int32}, length(x))
    nl = MD.NeighborList(x, unitcell, 2.5; skin=skin, nchunks=4, ntasks=2)
    MD.build!(nl, x, images)
    rng = Xoshiro(6)

    # Small moves keep the list valid, even for particles leaving the box
    for _ in 1:5
        x .+= [0.035 * normalize(randn(rng, SVector{3,Float64})) for _ in x]
    end
    @test !MD.needs_rebuild(nl, x)
    @test !MD.update!(nl, x, images)
    (E0, W0, F0) = brute_force(x, d, unitcell, pot, 2.5)
    F = similar(x)
    (E, W) = MD.compute_forces!(F, x, d, pot, nl)
    @test E ≈ E0 rtol = 1e-12
    @test max_force_error(F, F0) < 1e-10 * maximum(norm, F0)

    # Moving one particle by more than skin / 2 triggers a rebuild that wraps positions
    x[1] += SVector(-0.6 * skin - x[1][1], 0.0, 0.0)
    unwrapped = [x[i] + unitcell * images[i] for i in eachindex(x)]
    nbuilds = nl.nbuilds
    @test MD.update!(nl, x, images)
    @test nl.nbuilds == nbuilds + 1
    @test all(all(0 .<= inv(unitcell) * xi .< 1) for xi in x)
    @test all(x[i] + unitcell * images[i] ≈ unwrapped[i] for i in eachindex(x))
    (E0, W0, F0) = brute_force(x, d, unitcell, pot, 2.5)
    (E, W) = MD.compute_forces!(F, x, d, pot, nl)
    @test E ≈ E0 rtol = 1e-12
    @test max_force_error(F, F0) < 1e-10 * maximum(norm, F0)

    @test_throws ArgumentError MD.NeighborList(x, unitcell, 2.5; skin=-0.1)
    @test_throws ArgumentError MD.NeighborList(x, unitcell, 0.0)
end

@testset "State forces with default chunks and tasks" begin
    # Regression: the first force evaluation used to return zero, and threaded runs
    # multiplied forces by the number of threads
    (state, params, L) = lj_state(5; jitter=0.3, seed=8)
    MD.compute_forces!(state, params.potential)
    (E0, W0, F0) = brute_force(
        state.positions, state.diameters, state.unitcell, params.potential, 2.5
    )
    @test state.energy ≈ E0 rtol = 1e-12
    @test state.virial ≈ W0 rtol = 1e-12
    @test max_force_error(state.forces, F0) < 1e-10 * maximum(norm, F0)
end
