#!/usr/bin/env bash
# LJ melt benchmark: LAMMPS (MPI ranks) vs MolecularDynamics.jl (threads), same protocol.
# Usage: ./run.sh [ncell=6] [nsteps=10000] [cores=1]   (N = 4 ncell^3 particles)
set -euo pipefail
cd "$(dirname "$0")"
ncell=${1:-6}
nsteps=${2:-10000}
cores=${3:-1}

mkdir -p lmp_out
mpirun -np "$cores" --bind-to core lmp -in in.melt -var ncell "$ncell" -var nsteps "$nsteps" \
    -var outdir lmp_out -log lmp_out/log.lammps -screen none
echo "LAMMPS:              $(grep 'Loop time' lmp_out/log.lammps)"
echo -n "MolecularDynamics.jl: "
julia --project=.. -t "$cores" bench_md.jl "$ncell" "$nsteps"
