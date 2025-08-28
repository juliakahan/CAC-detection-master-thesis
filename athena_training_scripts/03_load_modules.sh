#!/bin/bash
# Load necessary compiler and MPI modules on cluster

module purge
module load GCC/${GCC_VERSION:-12.3.0} \
            OpenMPI/${OPENMPI_VERSION:-4.1.5} \
            mpi4py/${MPI4PY_VERSION:-3.1.4}
