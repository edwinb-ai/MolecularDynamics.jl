# MolecularDynamics.jl

A simple molecular dynamics code that samples the canonical ensemble ($`NVT`$), the
microcanonical ensemble ($`NVE`$) and also perform Brownian dynamics simulations in the ($`NVT`$)
ensemble.

### Example

It can be used in as a module, here is a simple example script.

```julia
using Printf
using MolecularDynamics

function main()
    # Define some thermodynamic variables
    packing_fraction = 0.47
    density = 6.0 * packing_fraction / pi
    ktemp = 1.4737
    n_particles = 2^10
    println("Number of particles: $(n_particles)")
    dt = 0.001
    # Instantiate a `Parameters` object to hold this information
    params = Parameters(density, n_particles, dt)

    # Create a directory to save all the files, this will be the root directory
    pathname = joinpath(
        @__DIR__, "test_N=$(n_particles)_density=$(@sprintf("%.4g", density))"
    )
    mkpath(pathname)

    # We create a thermostat, and the second argument is the damping
    thermostat = NVT(ktemp, 100.0 * dt)
    # Define an array of diameters, a monodisperse system is one sigma
    diameters = ones(n_particles)
    # Here we initialize the state of the simulation
    state = initialize_state(params, pathname, diameters; random_init=true)
    # Velocities have to be explicitly set
    init_temperature = initial_temperature_for_velocities(ktemp)
    velocities = initialize_velocities(init_temperature, rng, params.n_particles, dimension)
    state.velocities = velocities
    
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
        traj_name="production.xyz",
        thermo_name="production_thermo.txt",
        log_times=false,
        compress=true,
    )

    return nothing
end

main()
```

## How to add different potentials

To add a user-defined interaction potential we have to overload the `evaluate` method, and create
a special sub-type of the `Potential` type. Here is a commented example script for a polydisperse
mixture that reads in a configuration file, and sets up the interaction potential.

```julia
using MolecularDynamics
using Printf: @sprintf
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

""" This function will evaluate the potential, but note that the first four arguments have
to be exactly these ones: `pot` the type of the potential, always has to be the same
type as we defined earlier. Then `r` is the distance, which will be computed by the
simulation code. And we always pass two diameters, for the distance between the two particles
that we are dealing with, and the names have to be exactly `sigma1` and `sigma2`.
The keyword arguments can be anything you need for the potential to work properly, in this case
I need the cutoff radius and the value of the non-additivity of the particles.
"""
function evaluate(
    pot::Polydisperse,
    r::Real;
    sigma1::Real,
    sigma2::Real;
    rcut::Real=1.25,
    non_additivity::Real=0.2,
)
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
    # Instantiate a `Parameters` object to hold this information
    phs = Polydisperse()
    params = Parameters(density, n_particles, dt, phs)

    # Create a directory to save all the files, this will be the root directory
    pathname = joinpath(
        @__DIR__, "poly_2D_N=$(n_particles)_density=$(@sprintf("%.4g", density))"
    )
    mkpath(pathname)

    # Here we initialize the state of the simulation from a file
    state = initialize_state(
        params, pathname; dimension=2, from_file="snapshot_step_10000000.xyz"
    )
    # Velocities have to be explicitly set
    init_temperature = initial_temperature_for_velocities(ktemp)
    velocities = initialize_velocities(init_temperature, rng, params.n_particles, dimension)
    state.velocities = velocities
    # We want to simulation standard NVE
    run_simulation!(state, params, NVE(), 100_000, 1_000, pathname; compress=true)

    return nothing
end

main()
```

## Features

- Uses the Bussi-Donadio-Parrinello thermostat to control temperature.
- Integrates particles' positions and velocities using velocity Verlet.
- The Brownian dynamics integrator is a simple Euler-Murayama first order integrator. This is essentially the approach of the Ermak-McCammon algorithm. The only difference is that a uniform distribution with the same moments as a normal distribution is sampled; this is done for efficiency of the code.
- Can handle very large systems thanks to the cell implementation of [CellListMap.jl](https://github.com/m3g/CellListMap.jl)
- For now it can compute energy and pressure, but also outputs the trajectory of the simulation for post-processing.
- The Lennard-Jones potential and a pseudo hard sphere potential are implemented. Switching between them requires you to modify the source code. Long range corrections for the Lennard-Jones potential are included.
  - Benchmarks against LAMMPS and NIST results for the Lennard-Jones system are in the [wiki](https://github.com/edwinb-ai/MolecularDynamics.jl/wiki/Lennard%E2%80%90Jones-results).
- Initial configurations can be created in a random configuration. Random configurations are then packed (removing overlaps) using [Packmol.jl](https://github.com/m3g/Packmol.jl).
- Now it can save configurations using XYZ and LAMMPS format, but one cannot choose it. Trajectories are saved in Extended XYZ format, and compressed with `zstd` after the full trajectory has been written.
    - It can also print the unwrapped coordinates of the particles, which are useful for the analysis of dynamical properties. However, the only format that support this is the LAMMPS format.
- The configuration can now be minimized to an local energy minimum with the fast inertial relaxation engine (FIRE) algorithm.

## TODO
- Also, the configuration of the system is always at random and packed, which helps to start a random simulation. However, the user should be able to set their configuration as they want, and the code do the integration of the equations of motion.
    - I think that for this the code is generic enough that one should be able to pass an array of positions to the state. Right now the state does not accept this, but it would be useful if the user can pass a configuration of their choosing, and the engine will just integrate the equations of motion. This will also reduce the amount of dependencies that we need to care of.
