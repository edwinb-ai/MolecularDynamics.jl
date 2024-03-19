# MolecularDynamics.jl

A simple molecular dynamics code that samples the canonical ensemble ($`NVT`$).

## Features

- Uses the Bussi-Donadio-Parrinello thermostat to control temperature.
- Integrates particles' positions and velocities using velocity Verlet.
- Can handle very large systems thanks to the fantastic cell implementation of [CellListMap.jl](https://github.com/m3g/CellListMap.jl)
- For now it can compute energy and pressure, but also outputs the trajectory of the simulation for post-processing.

## TODO
- Introduce structures to handle polydispersity.
- Add other output formats (LAMMPS, Extended XYZ, etc.).
- Add instructions on how to use it. For now, it is not really usable unless you dig into the code and modify it yourself.
