# MolecularDynamics.jl

A simple molecular dynamics code that samples the canonical ensemble ($`NVT`$).

## Features

- Uses the Bussi-Donadio-Parrinello thermostat to control temperature.
- Integrates particles' positions and velocities using velocity Verlet.
- Can handle very large systems thanks to the cell implementation of [CellListMap.jl](https://github.com/m3g/CellListMap.jl)
- For now it can compute energy and pressure, but also outputs the trajectory of the simulation for post-processing.
- The Lennard-Jones potential and a pseudo hard sphere potential are implemented. Switching between them requires you to modify the source code. Long range corrections for the Lennard-Jones potential are included.
  - Benchmarks against LAMMPS and NIST results for the Lennard-Jones system are in the [wiki](https://github.com/edwinb-ai/MolecularDynamics.jl/wiki/Lennard%E2%80%90Jones-results).
- Initial configurations can be created as a simple cubic and also in a random configuration. Random configurations are then packed (removing overlaps) using [Packmol.jl](https://github.com/m3g/Packmol.jl).

## TODO
- Introduce structures to handle polydispersity.
- Add other output formats (LAMMPS, Extended XYZ, etc.).
- Add instructions on how to use it. For now, it is not really usable unless you dig into the code and modify it yourself.
