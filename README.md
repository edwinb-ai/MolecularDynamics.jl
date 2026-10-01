# MolecularDynamics.jl

A simple molecular dynamics code that samples the canonical ensemble ($`NVT`$), the
microcanonical ensemble ($`NVE`$) and also perform Brownian dynamics simulations in the ($`NVT`$)
ensemble.

### Example

It can be used as a module, here is a simple example script.

```julia
using Printf
using Random
using MolecularDynamics

function main()
    # Define some thermodynamic variables
    packing_fraction = 0.47
    density = 6.0 * packing_fraction / pi
    ktemp = 1.4737
    n_particles = 2^10
    println("Number of particles: $(n_particles)")
    dt = 0.001
    dimension = 3
    rng = Random.Xoshiro(1234)
    # Instantiate a `Parameters` object to hold this information, here for pseudo hard spheres
    params = Parameters(density, n_particles, dt, PseudoHS())

    # Create a directory to save all the files, this will be the root directory
    pathname = joinpath(
        @__DIR__, "test_N=$(n_particles)_density=$(@sprintf("%.4g", density))"
    )
    mkpath(pathname)

    # We create a thermostat, and the second argument is the damping
    thermostat = NVT(ktemp, 100.0 * dt)
    # Random configuration of unit-diameter particles, packed to remove overlaps. The
    # cutoff has to cover the range of the potential, which is about 1.02 diameters here.
    state = initialize_state(
        params, pathname; dimension=dimension, random_init=true, cutoff=1.1, rng=rng
    )
    # Velocities have to be explicitly set
    init_temperature = initial_temperature_for_velocities(ktemp)
    state.velocities = initialize_velocities(
        init_temperature, rng, params.n_particles, dimension
    )

    # We run the simulation for 1_000_000 time steps, and we print data
    # every 100_000 time steps
    # The `compress=true` enables compression of the trajectory files using `zstd`
    run_simulation!(state, params, thermostat, 1_000_000, 100_000, pathname; compress=true)

    # Now we do NVE
    run_simulation!(
        state,
        params,
        NVE(),
        1_000_000,
        100_000,
        pathname;
        traj_name="production.lammpstrj",
        thermo_name="production_thermo.txt",
        log_times=false,
        compress=true,
    )

    return nothing
end

main()
```

Run it with several threads to use more cores, e.g. `julia -t 8 script.jl`.

## How to add different potentials

To add a user-defined interaction potential we have to overload the `evaluate` method, and create
a special sub-type of the `Potential` type. Here is a commented example script for a polydisperse
mixture that reads in a configuration file, and sets up the interaction potential.

Optionally, a potential can also overload `evaluate_r2(pot, r2, sigma1, sigma2)`, which
receives the squared distance and returns the energy and the force divided by the distance.
This avoids a square root and a division per pair; see `LennardJones` for an example.

```julia
using MolecularDynamics
using Printf: @sprintf
using Random
using FastPow: @fastpow
# IMPORTANT: always `import` to overload
import MolecularDynamics: Potential, evaluate

# This type will define the new interaction potential, but needs this form
struct Polydisperse{F<:Function} <: Potential
    potf::F
end

# The function has to always return a pair of values, the energy and the force
# evaluated. This can have as many argument as needed, as long as the return values
# are consistent.
@fastpow function poly_potential(r, sigma, r_cut)
    uij = 0.0
    fij = 0.0

    # This is the potential energy for the polydisperse potential
    if r < r_cut * sigma
        term_1 = (sigma / r)^12
        c0 = -28.0 / (r_cut^12)
        c2 = 48.0 / (r_cut^14)
        c4 = -21.0 / (r_cut^16)
        term_2 = c2 * (r / sigma)^2
        term_3 = c4 * (r / sigma)^4
        uij = term_1 + c0 + term_2 + term_3
    else
        uij = 0.0
    end

    # This is the force evaluation, the virial is computed in the `MolecularDynamics.jl` code
    if r < r_cut * sigma
        c2 = 48.0 / (r_cut^14)
        c4 = -21.0 / (r_cut^16)
        fij = 12.0 * sigma^12 / r^13 - 2.0 * c2 * r / (sigma^2) - 4.0 * c4 * r^3 / (sigma^4)

    else
        fij = 0.0
    end

    # Always return this pair of values, in this order
    return uij, fij
end

# This is a constructor, and it is saying, whenever you create an object
# `Polydisperse`, assign the `poly_potential` function to it.
Polydisperse() = Polydisperse(poly_potential)

""" This function will evaluate the potential. The simulation code calls it with exactly
these four positional arguments: `pot`, the potential type we defined earlier; `r`, the
distance between two particles; and `sigma1` and `sigma2`, the diameters of those two
particles. It must return the energy and the force, `-dU/dr`. Any other parameter of the
potential, here the cutoff radius and the non-additivity, is set inside the function or
stored in the `Polydisperse` type.
"""
function evaluate(pot::Polydisperse, r::Real, sigma1::Real, sigma2::Real)
    rcut = 1.25
    non_additivity = 0.2
    # We need to compute the special non-additive sigma
    σ_eff = 0.5 * (sigma1 + sigma2)
    σ_eff *= (1.0 - non_additivity * abs(sigma1 - sigma2))

    return pot.potf(r, σ_eff, rcut)
end

function main()
    # Define some thermodynamic variables
    density = 1.0
    ktemp = 0.11
    n_particles = 1200
    println("Number of particles: $(n_particles)")
    dt = 0.005
    dimension = 2
    rng = Random.Xoshiro(1234)
    # Instantiate a `Parameters` object to hold this information
    phs = Polydisperse()
    params = Parameters(density, n_particles, dt, phs)

    # Create a directory to save all the files, this will be the root directory
    pathname = joinpath(
        @__DIR__, "poly_2D_N=$(n_particles)_density=$(@sprintf("%.4g", density))"
    )
    mkpath(pathname)

    # Here we initialize the state of the simulation from a file. The cutoff has to cover
    # the range of the potential, 1.25 times the largest σ_eff.
    state = initialize_state(
        params,
        pathname;
        dimension=dimension,
        from_file="snapshot_step_10000000.xyz",
        cutoff=2.0,
        rng=rng,
    )
    # Velocities have to be explicitly set
    init_temperature = initial_temperature_for_velocities(ktemp)
    state.velocities = initialize_velocities(
        init_temperature, rng, params.n_particles, dimension
    )
    # We want to simulation standard NVE
    run_simulation!(state, params, NVE(), 100_000, 1_000, pathname; compress=true)

    return nothing
end

main()
```

## Vibrational modes

The Hessian of a (minimized) configuration is computed exactly, without finite differences:
ForwardDiff differentiates each pair energy through your `evaluate` method, which therefore
has to accept `r::Real` rather than only `r::Float64`.

```julia
minimize!(state, params, pathname, dimension; tol=1e-10)
H = hessian(state, params)              # sparse dN × dN matrix
(ω, modes) = lowest_modes(H, 100)       # lowest 100 modes, for large systems
(ω, modes) = normal_modes(H)            # all modes, up to a few thousand particles
P = participation_ratio(modes, dimension)
```

Masses are 1, frequencies are `sqrt` of the eigenvalues (negative for unstable directions),
and the participation ratio is about 1 for extended modes and of order `1/N` for localized
ones. See `examples/vibrational_modes.jl`.

## Features

- Uses the Bussi-Donadio-Parrinello thermostat to control temperature.
- Integrates particles' positions and velocities using velocity Verlet.
- The Brownian dynamics integrator is a simple Euler-Maruyama first order integrator. This is essentially the approach of the Ermak-McCammon algorithm. The only difference is that a uniform distribution with the same moments as a normal distribution is sampled; this is done for efficiency of the code.
- Forces are computed with a Verlet neighbor list built from cell lists, rebuilt only when a particle has moved more than half the skin (`skin` keyword of `initialize_state`, default `0.3`). It supports 2D and 3D, orthorhombic and triclinic boxes, and any box size relative to the cutoff.
- Runs in parallel with Julia threads, e.g. `julia -t 8 script.jl`. On a Lennard-Jones melt it matches or beats LAMMPS on the same number of cores; see `benchmark/run.sh` to reproduce.
- For now it can compute energy and pressure, but also outputs the trajectory of the simulation for post-processing.
- The Lennard-Jones potential (optionally energy- or force-shifted, and with long range corrections), a Lennard-Jones potential with an XPLOR switching function, and a pseudo hard sphere potential are implemented; the potential is chosen when creating `Parameters`. Generic user-defined interaction potentials can be defined as shown above.
  - Benchmarks against LAMMPS and NIST results for the Lennard-Jones interaction potential are in the [wiki](https://github.com/edwinb-ai/MolecularDynamics.jl/wiki/Lennard%E2%80%90Jones-results).
- Initial configurations can be random, read from an extended XYZ file (`from_file`), or given directly with the `positions`, `diameters` and `unitcell` keywords of `initialize_state`. Random configurations are packed (removing overlaps) using [Packmol.jl](https://github.com/m3g/Packmol.jl).
- Configurations (`init.xyz`, `final.xyz`, `minimized.xyz`) are saved in extended XYZ format, readable by OVITO and ASE; 2D systems are written with `z = 0` and `pbc="T T F"`. Trajectories and snapshots are saved in the LAMMPS dump format, whatever their file name, and can be compressed with `zstd` after the full trajectory has been written.
    - The LAMMPS dumps include the unwrapped coordinates of the particles, which are useful for the analysis of dynamical properties.
- The configuration can be minimized to a local energy minimum with the fast inertial relaxation engine (FIRE) algorithm.

## Changes between versions

See [CHANGELOG.md](CHANGELOG.md), which also has the notes for upgrading between versions.

## Running the tests

```shell
julia --project -e 'using Pkg; Pkg.test()'
julia --project -e 'using Pkg; Pkg.test(julia_args=["--threads=4"])'
```
