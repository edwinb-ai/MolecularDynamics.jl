# Central finite difference of the energy
function numerical_force(pot, r, s1, s2; h=1e-6)
    return -(first(evaluate(pot, r + h, s1, s2)) - first(evaluate(pot, r - h, s1, s2))) /
           (2h)
end

@testset "Potentials" begin
    @testset "force is -dU/dr: $name" for (name, pot, sigmas, rs) in [
        ("LennardJones", LennardJones(; r_cut=2.5), [(1.0, 1.0), (0.8, 1.3)], 0.9:0.1:2.4),
        (
            "LennardJones shifted",
            LennardJones(; r_cut=2.5, shift=true),
            [(1.0, 1.0), (0.8, 1.3)],
            0.9:0.1:2.4,
        ),
        (
            "LennardJones force-shifted",
            LennardJones(; r_cut=2.5, force_shift=true),
            [(1.0, 1.0), (0.8, 1.3)],
            0.9:0.1:2.4,
        ),
        (
            "LennardJonesXPLOR",
            LennardJonesXPLOR(1.0, 1.0, 2.0, 2.5, false),
            [(1.0, 1.0), (0.9, 1.2)],
            0.9:0.05:2.45,
        ),
        ("PseudoHS", PseudoHS(), [(1.0, 1.0), (0.8, 1.2), (1.5, 1.5)], nothing),
        ("Gaussian (user-defined)", Gaussian(), [(1.0, 1.0)], 0.1:0.2:3.0),
    ]
        for (s1, s2) in sigmas
            σ = (s1 + s2) / 2
            radii = rs === nothing ? range(0.97σ, 1.019σ; length=9) : rs
            for r in radii
                (u, f) = evaluate(pot, r, s1, s2)
                @test f ≈ numerical_force(pot, r, s1, s2) rtol = 1e-5 atol = 1e-8
                # The sqrt-free path agrees with `evaluate`
                (u2, f_over_r) = MD.evaluate_r2(pot, r^2, s1, s2)
                @test u2 ≈ u rtol = 1e-12
                @test f_over_r * r ≈ f rtol = 1e-12
            end
        end
    end

    @testset "Lennard-Jones" begin
        pot = LennardJones(; r_cut=2.5)
        # Force vanishes at the minimum, energy is -ε there
        (u, f) = evaluate(pot, 2^(1 / 6), 1.0, 1.0)
        @test u ≈ -1.0
        @test abs(f) < 1e-12
        @test evaluate(pot, 2.5, 1.0, 1.0) == (0.0, 0.0)
        @test MD.evaluate_r2(pot, 2.5^2, 1.0, 1.0) == (0.0, 0.0)
        # Mixed diameters use the arithmetic mean
        @test evaluate(pot, 1.3, 0.8, 1.2) == evaluate(pot, 1.3, 1.0, 1.0)
    end

    @testset "Lennard-Jones shifts" begin
        rc = 2.5
        plain = LennardJones(; r_cut=rc)
        shifted = LennardJones(; r_cut=rc, shift=true)
        force_shifted = LennardJones(; r_cut=rc, force_shift=true)
        for (s1, s2) in ((1.0, 1.0), (0.8, 1.3))
            σ = (s1 + s2) / 2
            (Vcut, Fcut) = MD.lj_cut_values(1.0, σ, rc)
            for r in 0.9:0.2:2.3
                (u0, f0) = evaluate(plain, r, s1, s2)
                # Energy shift only moves the energy
                @test evaluate(shifted, r, s1, s2) == (u0 - Vcut, f0)
                # Force shift makes both vanish linearly at the cutoff
                @test evaluate(force_shifted, r, s1, s2)[1] ≈ u0 - Vcut + (r - rc) * Fcut
                @test evaluate(force_shifted, r, s1, s2)[2] ≈ f0 - Fcut
            end
            for pot in (shifted, force_shifted)
                @test abs(first(evaluate(pot, rc - 1e-10, s1, s2))) < 1e-9
                @test evaluate(pot, rc, s1, s2) == (0.0, 0.0)
                @test MD.evaluate_r2(pot, rc^2, s1, s2) == (0.0, 0.0)
            end
            @test abs(last(evaluate(force_shifted, rc - 1e-10, s1, s2))) < 1e-9
        end
        # The stored cutoff values are those of `sigma`
        @test (shifted.V_cut, shifted.F_cut) == MD.lj_cut_values(1.0, 1.0, rc)
    end

    @testset "XPLOR switching" begin
        pot = LennardJonesXPLOR(1.0, 1.0, 2.0, 2.5, false)
        lj = LennardJones(; r_cut=2.5)
        # Unchanged below r_on, smoothly zero at r_cut
        @test all(
            evaluate(pot, r, 1.0, 1.0) == evaluate(lj, r, 1.0, 1.0) for r in 0.9:0.1:1.9
        )
        (u, f) = evaluate(pot, 2.5 - 1e-9, 1.0, 1.0)
        @test abs(u) < 1e-12
        @test abs(f) < 1e-6
        @test evaluate(pot, 2.6, 1.0, 1.0) == (0.0, 0.0)
        # Keyword form kept for compatibility
        @test evaluate(pot, 1.5; sigma1=1.0, sigma2=1.0) == evaluate(pot, 1.5, 1.0, 1.0)
    end

    @testset "PseudoHS" begin
        pot = PseudoHS()
        for σ in (1.0, 1.4)
            @test first(evaluate(pot, σ, σ, σ)) ≈ 1.0
            # Continuous at the cutoff (the potential minimum)
            (u, f) = evaluate(pot, MD.b_param * σ * (1 - 1e-9), σ, σ)
            @test abs(u) < 1e-6
            @test abs(f) < 1e-4
            @test evaluate(pot, MD.b_param * σ * 1.001, σ, σ) == (0.0, 0.0)
        end
    end

    @testset "Long-range corrections" begin
        (rc, ρ) = (2.5, 0.8)
        # Tail of the LJ energy per particle: 2πρ ∫ u(r) r² dr from rc to ∞
        u_lj(r) = 4 * (r^-12 - r^-6)
        rs = range(rc, 200.0; length=2_000_001)
        integral =
            sum(u_lj(r) * r^2 for r in rs) * step(rs) -
            (u_lj(rc) * rc^2 + u_lj(200.0) * 200.0^2) * step(rs) / 2
        # Analytic remainder beyond r = 200
        integral += 4 * (200.0^-9 / 9 - 200.0^-3 / 3)
        @test MD.ener_lrc(rc, ρ) ≈ 2π * ρ * integral rtol = 1e-6
        on = LennardJones(; r_cut=rc, tail_correction=true)
        off = LennardJones(; r_cut=rc)
        @test MD.energy_lrc(on, 100, 125.0) ≈ 100 * MD.ener_lrc(rc, 0.8)
        @test MD.energy_lrc(off, 100, 125.0) == 0.0
        @test MD.pressure_lrc(on, 100, 125.0) ≈ MD.pressure_lrc(rc, 0.8)
        @test MD.pressure_lrc(off, 100, 125.0) == 0.0
        @test MD.energy_lrc(Gaussian(), 100, 125.0) == 0.0
    end

    struct NotImplemented <: Potential end
    @test_throws ErrorException evaluate(NotImplemented(), 1.0, 1.0, 1.0)
end
