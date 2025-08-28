#!/bin/bash
#SBATCH -J "${JOB_NAME:-plan_dataset}"
#SBATCH -A "${ACCOUNT:-default-account}"
#SBATCH -p "${PARTITION:-gpu}"
#SBATCH -N 1
#SBATCH --cpus-per-task="${CPUS_PER_TASK:-8}"
#SBATCH --mem="${MEMORY:-32G}"
#SBATCH -t "${TIME_LIMIT:-02:00:00}"
#SBATCH -o "${JOB_NAME:-plan_dataset}_%j.out"
#SBATCH -e "${JOB_NAME:-plan_dataset}_%j.err"

# modules and environment
module purge
module load GCC/${GCC_VERSION:-12.3.0} OpenMPI/${OPENMPI_VERSION:-4.1.5} Python/${PYTHON_VERSION:-3.11.3}

# Activate Python environment
VENV_PATH="${NNUNET_ENV:-$HOME/envs/nnunet}"
source "$VENV_PATH/bin/activate"

# nnU-Net paths
export nnUNet_raw="${NNUNET_RAW:-/path/to/nnUNet_raw}"
export nnUNet_preprocessed="${NNUNET_PREPROCESSED:-/path/to/nnUNet_preprocessed}"
export nnUNet_results="${NNUNET_RESULTS:-/path/to/nnUNet_results}"

echo "nnUNet_raw          = $nnUNet_raw"
echo "nnUNet_preprocessed = $nnUNet_preprocessed"
echo "nnUNet_results      = $nnUNet_results"
echo "Dataset ID          = ${DATASET_ID:-502} (Dataset${DATASET_ID:-502}_CAC)"

# Run nnU-Net preprocessing
nnUNetv2_plan_and_preprocess \
    -d "${DATASET_ID:-502}" \
    --verify_dataset_integrity \
    --multiprocessing

echo "Preprocessing finished: $nnUNet_preprocessed/Dataset${DATASET_ID:-502}_CAC"
