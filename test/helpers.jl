using MolecularDynamics
using StaticArrays
using LinearAlgebra
using Random
using DelimitedFiles: readdlm
using Logging: with_logger, NullLogger
import MolecularDynamics: Potential, evaluate

const MD = MolecularDynamics

# Smooth, bounded potential; defines only `evaluate`, so it exercises the `evaluate_r2` fallback
struct Gaussian <: Potential end
evaluate(::Gaussian, r::Real, s1::Real, s2::Real) = (exp(-r^2), 2r * exp(-r^2))

# Non-interacting particles
struct Ideal <: Potential end
evaluate(::Ideal, r::Real, s1::Real, s2::Real) = (0.0, 0.0)

"FCC lattice with `ncell^3` unit cells at number density `rho`; returns positions and box length."
function fcc_positions(ncell, rho)
    a = (4 / rho)^(1 / 3)
    basis = (
        SVector(0.0, 0.0, 0.0),
        SVector(0.5, 0.5, 0.0),
        SVector(0.5, 0.0, 0.5),
        SVector(0.0, 0.5, 0.5),
    )
    positions = [
        a * (SVector{3,Float64}(i, j, k) + b + SVector(0.25, 0.25, 0.25)) for
        i in 0:(ncell - 1) for j in 0:(ncell - 1) for k in 0:(ncell - 1) for b in basis
    ]
    return positions, a * ncell
end

"Energy, virial and forces summed over every periodic image within `cutoff` (O(N² Kᴰ))."
function brute_force(positions, diameters, unitcell, potential, cutoff; K=2)
    D = length(first(positions))
    N = length(positions)
    forces = zeros(SVector{D,Float64}, N)
    energy = 0.0
    virial = 0.0
    shifts = [
        unitcell * SVector{D,Float64}(Tuple(t)) for
        t in CartesianIndices(ntuple(_ -> (-K):K, D))
    ]
    for i in 1:N, j in i:N, shift in shifts
        (i == j && iszero(shift)) && continue
        r = positions[i] - positions[j] - shift
        d = norm(r)
        d < cutoff || continue
        (u, f) = evaluate(potential, d, diameters[i], diameters[j])
        # A particle and its own images: shifts ±t are the same pair
        weight = i == j ? 0.5 : 1.0
        energy += weight * u
        virial += weight * f * d
        forces[i] += f * r / d
        forces[j] -= f * r / d
    end
    return energy, virial, forces
end

"Run the package force routine on fresh copies and return (energy, virial, forces)."
function package_forces(
    positions, diameters, unitcell, potential, cutoff; skin=0.3, nchunks=1, ntasks=1
)
    x = copy(positions)
    images = zeros(SVector{length(first(x)),Int32}, length(x))
    nl = MD.NeighborList(x, unitcell, cutoff; skin=skin, nchunks=nchunks, ntasks=ntasks)
    MD.build!(nl, x, images)
    forces = similar(x)
    (energy, virial) = MD.compute_forces!(forces, x, diameters, potential, nl)
    return energy, virial, forces
end

max_force_error(F, G) = maximum(norm(F[i] - G[i]) for i in eachindex(F, G))

"LJ state on an FCC lattice (optionally jittered) with unit diameters."
function lj_state(
    ncell;
    rho=0.8442,
    jitter=0.0,
    cutoff=2.5,
    skin=0.3,
    seed=1,
    potential=LennardJones(; r_cut=cutoff),
)
    rng = Xoshiro(seed)
    (positions, L) = fcc_positions(ncell, rho)
    positions = [x + jitter * (rand(rng, SVector{3,Float64}) .- 0.5) for x in positions]
    N = length(positions)
    params = Parameters(rho, N, 0.005, potential)
    state = quiet() do
        return initialize_state(
            params,
            mktempdir();
            dimension=3,
            cutoff=cutoff,
            skin=skin,
            rng=rng,
            unitcell=[L, L, L],
            positions=positions,
            diameters=ones(N),
        )
    end
    return state, params, L
end

"Run `f` with logging disabled."
quiet(f) = with_logger(f, NullLogger())

"Read a thermo file written by `run_simulation!` as a matrix (step, energy, temperature, pressure)."
read_thermo(path) = readdlm(path; comments=true)
