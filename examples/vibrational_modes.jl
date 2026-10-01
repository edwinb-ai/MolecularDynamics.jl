# Vibrational modes of a minimized 2D configuration of the core-softened (quasicrystal
# forming) potential: quench, minimize, Hessian, lowest modes and participation ratios.
# `normal_modes(H)` gives the full spectrum instead, for up to a few thousand particles.
#
# Usage: julia --project=.. -t 8 vibrational_modes.jl [n_particles] [cooling_steps] [equilibration_steps] [n_modes]
using MolecularDynamics
using Random
using Printf
using DelimitedFiles: writedlm
import MolecularDynamics: Potential, evaluate

struct PseudoSS <: Potential end

# `r::Real` (not `r::Float64`) lets ForwardDiff differentiate the energy for the Hessian
function evaluate(::PseudoSS, r::Real, sigma1::Real, sigma2::Real)
    (k, delta) = (10.0, 1.35)
    uij = (sigma1 / r)^14 + 0.5 - 0.5 * tanh(k * (r - delta))
    fij = 14.0 * (sigma1 / r)^14 / r + 0.5 * k * sech(k * (r - delta))^2
    return uij, fij
end

function main(
    n_particles=1024, cooling_steps=50_000, equilibration_steps=50_000, n_modes=100
)
    (density, ktemp, dt, dimension) = (0.94, 0.16, 0.01, 2)
    rng = Random.Xoshiro(2024)
    # Switch the potential off smoothly between 1.8 and 2.0, so that the energy, force and
    # second derivative are continuous at the cutoff, as the low-frequency modes need
    potential = Smoothed(PseudoSS(); r_on=1.8, r_cut=2.0)
    params = Parameters(density, n_particles, dt, potential)
    pathname = mkpath(joinpath(@__DIR__, "modes_N=$(n_particles)"))

    state = initialize_state(
        params, pathname; dimension=dimension, cutoff=2.0, random_init=true, rng=rng
    )
    ramp = ExponentialRamp(1.0, ktemp, cooling_steps)
    state.velocities = initialize_velocities(
        initial_temperature_for_velocities(ramp), rng, n_particles, dimension
    )
    t_md = @elapsed run_simulation!(
        state,
        params,
        NVT(ramp, 100dt),
        cooling_steps + equilibration_steps,
        10_000,
        pathname,
    )

    # The low-frequency modes need a tightly converged minimum
    t_min = @elapsed (energy, converged) = minimize!(
        state,
        params,
        pathname,
        dimension;
        tol=1e-10,
        max_steps=1_000_000,
        dt_initial=0.001,
        dt_max=0.01,
    )
    t_hessian = @elapsed H = hessian(state, params)
    t_modes = @elapsed (ω, modes) = lowest_modes(H, n_modes)
    P = participation_ratio(modes, dimension)

    @printf(
        "N = %d  MD %.1f s, FIRE %.1f s (converged: %s, U/N = %.6f)\n",
        n_particles,
        t_md,
        t_min,
        converged,
        energy / n_particles
    )
    @printf(
        "Hessian %.3f s (%d × %d, %d non-zeros), lowest %d modes %.2f s\n",
        t_hessian,
        size(H)...,
        length(H.nzval),
        n_modes,
        t_modes
    )
    @printf(
        "zero modes: %d, negative modes: %d\n", count(abs.(ω) .< 1e-6), count(ω .< -1e-6)
    )
    println("lowest non-zero modes (ω, participation ratio):")
    for k in findall(ω .> 1e-6)[1:10]
        @printf("  %10.6f  %.4f\n", ω[k], P[k])
    end
    writedlm(joinpath(pathname, "modes.txt"), [ω P])
    return nothing
end

main(parse.(Int, ARGS)...)
