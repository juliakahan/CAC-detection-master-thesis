#!/bin/bash
# Load modules and activate Python venv with nnU-Net
module purge
module load GCC/12.3.0 OpenMPI/4.1.5 Python/3.11.3

# Activate virtual environment
source "${NNUNET_ENV:-$HOME/envs/nnunet}/bin/activate"

# Set nnU-Net directories
export nnUNet_raw="${NNUNET_RAW:-/path/to/nnUNet_raw}"
export nnUNet_preprocessed="${NNUNET_PREPROCESSED:-/path/to/nnUNet_preprocessed}"
export nnUNet_results="${NNUNET_RESULTS:-/path/to/nnUNet_results}"
