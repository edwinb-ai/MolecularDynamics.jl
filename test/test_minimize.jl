@testset "FIRE minimization" begin
    # A jittered FCC crystal relaxes back to the perfect lattice
    (state, params, L) = lj_state(3; jitter=0.1, seed=7)
    (lattice, _) = fcc_positions(3, 0.8442)
    N = params.n_particles
    (E_lattice, _, _) = brute_force(lattice, ones(N), state.unitcell, params.potential, 2.5)
    dir = mktempdir()
    (energy, converged) = quiet() do
        minimize!(state, params, dir, 3; tol=1e-8, max_steps=20_000)
    end
    @test converged
    @test energy / N ≈ E_lattice / N rtol = 1e-8
    @test sqrt(sum(f -> sum(abs2, f), state.forces) / (3 * (N - 1))) < 1e-8
    @test isfile(joinpath(dir, "minimized.xyz"))

    # Stopping early reports it
    (state, params, L) = lj_state(3; jitter=0.1, seed=7)
    (energy, converged) = quiet() do
        MD.fire_minimize!(state, params; max_steps=5)
    end
    @test !converged
    @test isfinite(energy)

    @test_throws ErrorException quiet() do
        minimize!(state, params, mktempdir(), 3; method=:steepest)
    end
end
