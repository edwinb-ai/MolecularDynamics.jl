using ForwardDiff: ForwardDiff

# Total energy over all periodic images as a function of the flattened coordinates
function total_energy(
    x::AbstractVector{T}, D, unitcell, diameters, potential, cutoff, K
) where {T}
    N = length(x) ÷ D
    positions = [SVector{D,T}(ntuple(a -> x[D * (i - 1) + a], D)) for i in 1:N]
    shifts = [
        unitcell * SVector{D,Float64}(Tuple(t)) for
        t in CartesianIndices(ntuple(_ -> (-K):K, D))
    ]
    energy = zero(T)
    for i in 1:N, j in i:N, shift in shifts
        (i == j && iszero(shift)) && continue
        r = positions[i] - positions[j] - shift
        d = sqrt(dot(r, r))
        d < cutoff || continue
        weight = i == j ? 0.5 : 1.0
        energy += weight * first(evaluate(potential, d, diameters[i], diameters[j]))
    end
    return energy
end

function make_state(
    positions, unitcell, potential, cutoff; diameters=ones(length(positions))
)
    D = size(unitcell, 1)
    params = Parameters(1.0, length(positions), 0.005, potential)
    state = quiet() do
        return initialize_state(
            params,
            mktempdir();
            dimension=D,
            cutoff=cutoff,
            unitcell=unitcell,
            positions=positions,
            diameters=diameters,
        )
    end
    return state, params
end

function reference_hessian(state, potential)
    D = state.dimension
    x = reduce(vcat, Vector.(state.positions))
    # Wrapped positions only need lattice shifts up to 1 + cutoff / (box height)
    h = minimum(1 / norm(inv(state.unitcell)[k, :]) for k in 1:D)
    K = 1 + floor(Int, state.neighbors.cutoff / h)
    U(y) = total_energy(
        y, D, state.unitcell, state.diameters, potential, state.neighbors.cutoff, K
    )
    return ForwardDiff.hessian(U, x)
end

# Potentials that `hessian` cannot use as intended
struct FloatOnly <: Potential end
evaluate(::FloatOnly, r::Float64, s1::Float64, s2::Float64) = (exp(-r^2), 2r * exp(-r^2))
struct WrongForce <: Potential end
evaluate(::WrongForce, r::Real, s1::Real, s2::Real) = (exp(-r^2), 4r * exp(-r^2))

@testset "Hessian" begin
    rng = Xoshiro(30)
    (fcc, L) = fcc_positions(2, 0.8442)
    jittered = [x + 0.1 * (rand(rng, SVector{3,Float64}) .- 0.5) for x in fcc]
    square = [
        SVector(i + 0.5, j + 0.5) * 1.1 + 0.1 * (rand(rng, SVector{2,Float64}) .- 0.5) for
        i in 0:3 for j in 0:3
    ]
    cases = [
        (
            "3D LJ, box smaller than twice the cutoff",
            jittered,
            SMatrix{3,3}(L * I(3)),
            LennardJones(; r_cut=2.5),
            2.5,
            ones(32),
        ),
        (
            "3D polydisperse force-shifted LJ",
            jittered,
            SMatrix{3,3}(L * I(3)),
            LennardJones(; r_cut=2.5, force_shift=true),
            2.5,
            0.9 .+ 0.2 .* rand(rng, 32),
        ),
        (
            "3D XPLOR",
            jittered,
            SMatrix{3,3}(L * I(3)),
            Smoothed(LennardJones(; r_cut=2.5); r_on=2.0, r_cut=2.5, switch=:xplor),
            2.5,
            ones(32),
        ),
        (
            "3D pseudo hard spheres",
            jittered,
            SMatrix{3,3}(L * I(3)),
            PseudoHS(),
            1.5,
            1.17 .+ 0.03 .* rand(rng, 32),
        ),
        (
            "2D triclinic, user-defined potential",
            square,
            SMatrix{2,2}([4.4 0.8; 0.0 4.4]),
            CoreSoftened(),
            2.0,
            ones(16),
        ),
        (
            "2D Gaussian, tiny box",
            square[1:6],
            SMatrix{2,2}([2.2 0.3; 0.0 2.0]),
            Gaussian(),
            2.5,
            ones(6),
        ),
    ]
    @testset "$name" for (name, positions, unitcell, potential, cutoff, diameters) in cases
        (state, params) = make_state(positions, unitcell, potential, cutoff; diameters)
        H = hessian(state, params)
        reference = reference_hessian(state, potential)
        @test size(H) == size(reference)
        @test maximum(abs.(Matrix(H) .- reference)) < 1e-9 * maximum(abs, reference)
        @test issymmetric(H)
        # Rigid translations cost no energy
        D = state.dimension
        for α in 1:D
            translation = [k % D == α % D ? 1.0 : 0.0 for k in 1:size(H, 1)]
            @test norm(H * translation) < 1e-9 * maximum(abs, reference)
        end
    end

    @testset "minimized crystal" begin
        # A relaxed triangular crystal is mechanically stable: d zero modes, the rest positive
        (nx, ny) = (6, 4)
        a = 1.12
        lattice = [
            SVector(a * (i + 0.5 * (j % 2)), a * sqrt(3) / 2 * j) for i in 0:(nx - 1) for
            j in 0:(2ny - 1)
        ]
        unitcell = SMatrix{2,2}([nx * a 0.0; 0.0 2ny * a * sqrt(3) / 2])
        positions = [x + 0.05 * (rand(rng, SVector{2,Float64}) .- 0.5) for x in lattice]
        (state, params) = make_state(
            positions, unitcell, LennardJones(; r_cut=2.5, force_shift=true), 2.5
        )
        (_, converged) = quiet() do
            MD.fire_minimize!(state, params; tol=1e-12, max_steps=100_000)
        end
        @test converged
        (ω, modes) = normal_modes(hessian(state, params))
        @test count(abs.(ω) .< 1e-5) == 2
        @test all(ω[3:end] .> 1e-3)
        @test all(participation_ratio(modes[:, 1:2], 2) .≈ 1.0)
        # Each eigenpair satisfies H v = ω² v
        H = hessian(state, params)
        @test norm(H * modes[:, end] - ω[end]^2 * modes[:, end]) < 1e-8 * ω[end]^2
    end

    @testset "lowest modes" begin
        (nx, ny) = (8, 5)
        a = 1.12
        lattice = [
            SVector(a * (i + 0.5 * (j % 2)), a * sqrt(3) / 2 * j) for i in 0:(nx - 1) for
            j in 0:(2ny - 1)
        ]
        unitcell = SMatrix{2,2}([nx * a 0.0; 0.0 2ny * a * sqrt(3) / 2])
        positions = [x + 0.05 * (rand(rng, SVector{2,Float64}) .- 0.5) for x in lattice]
        (state, params) = make_state(
            positions, unitcell, LennardJones(; r_cut=2.5, force_shift=true), 2.5
        )
        quiet() do
            MD.fire_minimize!(state, params; tol=1e-12, max_steps=100_000)
        end
        H = hessian(state, params)
        (ω_all, _) = normal_modes(H)
        (ω, modes) = lowest_modes(H, 12)
        @test size(modes) == (size(H, 1), 12)
        # Zero modes are only accurate to sqrt(eigenvalue error); the rest much better
        @test maximum(abs.(ω[1:2])) < 1e-5
        @test ω[3:end] ≈ ω_all[3:12] rtol = 1e-8
        @test all(norm(H * modes[:, k] - ω[k]^2 * modes[:, k]) < 1e-8 for k in 3:12)
        # Unstable configurations are rejected
        @test_throws ArgumentError lowest_modes(-H, 3)
        @test_throws ArgumentError lowest_modes(H, 0)
    end

    @testset "participation ratio" begin
        N = 10
        @test participation_ratio(normalize(ones(2N)), 2) ≈ 1.0
        localized = zeros(2N)
        localized[7] = 1.0
        @test participation_ratio(localized, 2) ≈ 1 / N
        @test participation_ratio(hcat(normalize(ones(2N)), localized), 2) ≈ [1.0, 1 / N]
    end

    @testset "potentials that ForwardDiff cannot use" begin
        (state, params) = make_state(square, SMatrix{2,2}(4.4 * I(2)), FloatOnly(), 2.0)
        @test_throws "r::Real" hessian(state, params)

        # A force that is not -dU/dr is reported
        (state, params) = make_state(square, SMatrix{2,2}(4.4 * I(2)), WrongForce(), 2.0)
        @test_logs (:warn, r"differs from -dU/dr") hessian(state, params)
        @test_logs hessian(state, params; check_forces=false)
    end
end
