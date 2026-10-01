# Jumps of U, dU/dr and d²U/dr² across r, from one-sided values
function derivative_jumps(pot, r; ε=1e-9)
    U(s) = first(evaluate(pot, s, 1.0, 1.0))
    dU(s) = ForwardDiff.derivative(U, s)
    d2U(s) = ForwardDiff.derivative(dU, s)
    return [abs(f(r + ε) - f(r - ε)) for f in (U, dU, d2U)]
end

@testset "Smoothed potentials" begin
    (r_on, r_cut) = (1.8, 2.0)

    @testset "switch functions" begin
        for switch in (MD.QuinticSwitch(), MD.XPLORSwitch())
            @test MD.switch_value(switch, r_on, r_on, r_cut)[1] ≈ 1.0
            @test abs(MD.switch_value(switch, r_on, r_on, r_cut)[2]) < 1e-12
            @test abs(MD.switch_value(switch, r_cut - 1e-12, r_on, r_cut)[1]) < 1e-12
            # dS/dr is the derivative of S
            for r in range(r_on, r_cut; length=7)[2:(end - 1)]
                S(s) = first(MD.switch_value(switch, s, r_on, r_cut))
                @test MD.switch_value(switch, r, r_on, r_cut)[2] ≈
                    ForwardDiff.derivative(S, r)
            end
        end
        # Only the quintic switch also has a vanishing second derivative at both ends
        d2S(switch, r) =
            ForwardDiff.derivative(s -> MD.switch_value(switch, s, r_on, r_cut)[2], r)
        @test abs(d2S(MD.QuinticSwitch(), r_on)) < 1e-10
        @test abs(d2S(MD.QuinticSwitch(), r_cut)) < 1e-10
        @test abs(d2S(MD.XPLORSwitch(), r_on)) > 1.0
    end

    @testset "continuity: $name" for (name, inner) in (
        ("Lennard-Jones", LennardJones(; r_cut=3.0)),
        ("core-softened", CoreSoftened()),
        ("Gaussian", Gaussian()),
    )
        quintic = Smoothed(inner; r_on=r_on, r_cut=r_cut)
        xplor = Smoothed(inner; r_on=r_on, r_cut=r_cut, switch=:xplor)
        for r in (r_on, r_cut)
            @test all(derivative_jumps(quintic, r) .< 1e-6)
            @test all(derivative_jumps(xplor, r)[1:2] .< 1e-6)
        end
        # Unchanged below r_on, zero from r_cut on
        for r in (0.9, 1.3, 1.7)
            @test evaluate(quintic, r, 1.0, 1.0) == evaluate(inner, r, 1.0, 1.0)
        end
        @test evaluate(quintic, r_cut, 1.0, 1.0) == (0.0, 0.0)
        @test evaluate(quintic, 2.7, 1.0, 1.0) == (0.0, 0.0)
        # Force is -dU/dr, and the r² path agrees, inside and outside the switching region
        for pot in (quintic, xplor), r in (1.2, 1.79, 1.85, 1.95)
            (u, f) = evaluate(pot, r, 1.0, 1.0)
            @test f ≈ numerical_force(pot, r, 1.0, 1.0) rtol = 1e-5 atol = 1e-8
            (u2, f_over_r) = MD.evaluate_r2(pot, r^2, 1.0, 1.0)
            @test u2 ≈ u rtol = 1e-12
            @test f_over_r * r ≈ f rtol = 1e-12
        end
    end

    @testset "XPLOR switch matches LennardJonesXPLOR" begin
        smoothed = Smoothed(LennardJones(; r_cut=2.5); r_on=2.0, r_cut=2.5, switch=:xplor)
        reference = LennardJonesXPLOR(1.0, 1.0, 2.0, 2.5, false)
        for r in 0.9:0.05:2.6
            @test all(evaluate(smoothed, r, 1.0, 1.0) .≈ evaluate(reference, r, 1.0, 1.0))
        end
    end

    @testset "argument checks" begin
        @test_throws ArgumentError Smoothed(Gaussian(); r_on=2.0, r_cut=1.8)
        @test_throws ArgumentError Smoothed(Gaussian(); r_on=1.8, r_cut=2.0, switch=:cubic)
        params = Parameters(1.0, 4, 0.005, Smoothed(Gaussian(); r_on=1.8, r_cut=2.0))
        positions = [
            SVector(1.0, 1.0), SVector(3.0, 1.0), SVector(1.0, 3.0), SVector(3.0, 3.0)
        ]
        @test_throws ArgumentError quiet() do
            initialize_state(
                params,
                mktempdir();
                dimension=2,
                cutoff=1.9,
                unitcell=[4.0, 4.0],
                positions=positions,
                diameters=ones(4),
            )
        end
        # Tail corrections of the wrapped potential do not apply
        pot = Smoothed(LennardJones(; tail_correction=true); r_on=2.2, r_cut=2.5)
        @test MD.energy_lrc(pot, 100, 125.0) == 0.0
    end
end
