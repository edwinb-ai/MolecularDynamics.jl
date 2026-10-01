# Changelog

Notable changes between versions of MolecularDynamics.jl. Versions follow semantic
versioning; while the version is below 1.0, a minor version (0.x) may break the API.

## [Unreleased]

### Added

- `hessian(state, params)`: exact sparse Hessian of the potential energy. ForwardDiff
  computes the radial derivatives of each pair energy through `evaluate`, so user-defined
  potentials need no extra code beyond accepting `r::Real`.
- `normal_modes(H)` (dense, all modes), `lowest_modes(H, k)` (shift-invert Lanczos, for
  large systems) and `participation_ratio(modes, dimension)`.
- `Smoothed(potential; r_on, r_cut, switch=:quintic)` switches any potential off smoothly
  between `r_on` and `r_cut`. The quintic switch keeps the energy, force and second
  derivative continuous; `switch=:xplor` keeps only the energy and force continuous.
  `initialize_state` rejects a cutoff shorter than `r_cut`.
- `examples/vibrational_modes.jl` for the 2D core-softened potential.

### Deprecated

- `LennardJonesXPLOR(ϵ, σ, r_on, r_cut, tail_correction)`: use the equivalent
  `Smoothed(LennardJones(; epsilon=ϵ, sigma=σ, r_cut, tail_correction); r_on, r_cut,
  switch=:xplor)`, or the default quintic switch, which also keeps the second derivative
  continuous. It still works, with a deprecation warning.

### Fixed

- The `LennardJones` tail corrections were missing the factor `ε σ³`, so they were only
  right for `epsilon = sigma = 1`.

### Changed

- The built-in potentials accept any `r::Real` in `evaluate`.
- `Smoothed` applies the tail corrections of the wrapped potential, if it has them.
- New dependencies: ForwardDiff, KrylovKit and SparseArrays.

## [0.8.2] - 2026-10-01

### Changed

- The upgrade notes moved from the README to this changelog.

## [0.8.1] - 2026-10-01

### Fixed

- 2D configurations (`init.xyz`, `final.xyz`, `minimized.xyz`) are written in standard
  extended XYZ: a 3×3 `Lattice`, three coordinates with `z = 0` and `pbc="T T F"`. 3D files
  also get `pbc="T T T"`. ASE could not read the previous 2D files. `read_file` still
  reads the old 2D layout, and now accepts any whitespace between columns.
- The README examples run with the current API.

## [0.8.0] - 2026-10-01

### Upgrading from 0.7

- `state.system` (the CellListMap particle system) has been replaced by `state.positions`,
  `state.velocities`, `state.forces`, `state.energy` and `state.virial`. Reading
  `state.system` still works but is deprecated.
- Positions, velocities and forces are vectors of `SVector`s: update an element with
  `x[i] = ...` instead of `x[i] .= ...`.
- During a run positions are only wrapped into the box when the neighbor list is rebuilt;
  they are wrapped again at the end of `run_simulation!` and `minimize!`, and
  `wrapped_positions(state)` returns wrapped copies at any time.
- `LennardJones(; shift=true)` and `LennardJones(; force_shift=true)` now shift the
  potential (they were ignored before).
- LAMMPS trajectories use the LAMMPS box format: orthogonal, restricted triclinic (correct
  tilt factors and bounds), or general triclinic (`abc origin`) for boxes that are not
  upper triangular.

### Added

- `state.system` returns the positions, forces, energy and virial with a deprecation
  warning, so scripts written for 0.7 that read it keep working.
- `fire_minimize!` accepts the `dimension` keyword again (it is ignored).

### Fixed

- `LennardJones` applies `shift` and `force_shift`, computed for the mixed σ of each pair.
  The force-shifted energy had the wrong sign in its linear term.
- LAMMPS dumps of triclinic boxes had swapped `xz` and `yz` tilts and used the lengths of
  the box vectors as bounds. 2D dumps now use the LAMMPS header and span z from -0.5
  to 0.5.

## [0.7.1] - 2026-10-01

### Changed

- Forces are computed with a Verlet neighbor list built from cell lists, rebuilt only
  when a particle has moved more than half the skin (`skin` keyword of
  `initialize_state`, default `0.3`), instead of rebuilding CellListMap cell lists every
  step. It supports 2D and 3D, triclinic boxes, and any box size relative to the cutoff,
  and runs in parallel with Julia threads. On a Lennard-Jones melt this is about 4 times
  faster on one core and matches or beats LAMMPS on 1 and 8 cores.
- Positions, velocities and forces are stored as `SVector`s, and `state.system` was
  removed (read access restored in 0.8.0). CellListMap is no longer a dependency.
- Potentials may overload `evaluate_r2(pot, r2, sigma1, sigma2)` to skip the square
  root; `LennardJones` does.

### Added

- A test suite (`Pkg.test()`), and a benchmark against LAMMPS in `benchmark/`.
- `wrapped_positions(state)`.

### Fixed

- Forces were multiplied by the number of threads in multithreaded runs.
- The first velocity Verlet steps used zero forces.
- Brownian dynamics called undefined functions and shared noise between threads.
- FIRE made all velocities the same object after a reset; `minimize!` now returns
  `(energy, converged)`.
- `LennardJonesXPLOR` could not be evaluated, and its force was wrong.
- `PseudoHS` force and cutoff for diameters other than 1.
- Starting from a configuration file (`from_file`) crashed.

## Earlier versions

See the git history.
