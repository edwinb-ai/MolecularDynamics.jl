# LJ FCC -> liquid melt with MolecularDynamics.jl, same protocol as `in.melt` (LAMMPS).
# Usage: julia --project=.. -t <threads> bench_md.jl <ncell> <nsteps> [skin]
using MolecularDynamics
using StaticArrays
using Random
using Printf
using Logging: with_logger, NullLogger

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

function setup(ncell, skin, outdir)
    rho = 0.8442
    (positions, L) = fcc_positions(ncell, rho)
    N = length(positions)
    rng = Xoshiro(87287)
    params = Parameters(rho, N, 0.005, LennardJones(; r_cut=2.5))
    state = with_logger(NullLogger()) do
        return initialize_state(
            params,
            outdir;
            cutoff=2.5,
            skin=skin,
            rng=rng,
            unitcell=[L, L, L],
            positions=positions,
            diameters=ones(N),
        )
    end
    state.velocities = initialize_velocities(2.0, rng, N, 3)
    return state, params
end

function main(ncell, nsteps, skin)
    outdir = mkpath(joinpath(@__DIR__, "jl_out", "n$(ncell)_t$(Threads.nthreads())"))
    thermostat = NVT(2.0, 0.5)
    # Warm-up run to compile
    (state, params) = setup(ncell, skin, outdir)
    run_simulation!(state, params, thermostat, 20, 10, outdir)
    # Timed run
    (state, params) = setup(ncell, skin, outdir)
    t = @timed run_simulation!(state, params, thermostat, nsteps, 1000, outdir)
    N = params.n_particles
    @printf(
        "N=%d threads=%d steps=%d skin=%.2f  time=%.3f s  %.3f ms/step  %.3f Matom-step/s  builds=%d  alloc=%.1f MiB  gc=%.1f%%\n",
        N,
        Threads.nthreads(),
        nsteps,
        skin,
        t.time,
        1e3 * t.time / nsteps,
        N * nsteps / t.time / 1e6,
        state.neighbors.nbuilds,
        t.bytes / 2^20,
        100 * t.gctime / t.time
    )
    return nothing
end

main(
    parse(Int, ARGS[1]),
    parse(Int, ARGS[2]),
    length(ARGS) > 2 ? parse(Float64, ARGS[3]) : 0.3,
)
