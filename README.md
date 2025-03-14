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
    state = initialize_state(params, ktemp, pathname, diameters; random_init=true)
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

## Features

- Uses the Bussi-Donadio-Parrinello thermostat to control temperature.
- Integrates particles' positions and velocities using velocity Verlet.
- The Brownian dynamics integrator is a simple Euler-Murayama first order integrator. This is essentially the approach of the Ermak-McCammon algorithm. The only difference is that a uniform distribution with the same moments as a normal distribution is sampled; this is done for efficiency of the code.
- Can handle very large systems thanks to the cell implementation of [CellListMap.jl](https://github.com/m3g/CellListMap.jl)
- For now it can compute energy and pressure, but also outputs the trajectory of the simulation for post-processing.
- The Lennard-Jones potential and a pseudo hard sphere potential are implemented. Switching between them requires you to modify the source code. Long range corrections for the Lennard-Jones potential are included.
  - Benchmarks against LAMMPS and NIST results for the Lennard-Jones system are in the [wiki](https://github.com/edwinb-ai/MolecularDynamics.jl/wiki/Lennard%E2%80%90Jones-results).
- Initial configurations can be created in a random configuration. Random configurations are then packed (removing overlaps) using [Packmol.jl](https://github.com/m3g/Packmol.jl).
- Now it can save configurations using XYZ and LAMMPS format, but one cannot choose it. Trajectories are saved in Extended XYZ format, and compressed with `zstd`.

## TODO
- Make the user change the interaction potential. Right now, only the pseudo hard-sphere and Lennard-Jones are implemented.
- Introduce structures to handle polydispersity.
