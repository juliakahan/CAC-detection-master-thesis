#!/bin/bash
#SBATCH -J "${JOB_NAME:-nnunet_train}"
#SBATCH -A "${ACCOUNT:-default-account}"
#SBATCH -p "${PARTITION:-gpu}"
#SBATCH --gres="gpu:${GPU_TYPE:-a100}:${NUM_GPUS:-1}"
#SBATCH --cpus-per-task="${CPUS_PER_TASK:-8}"
#SBATCH --mem="${MEMORY:-48G}"
#SBATCH -t "${TIME_LIMIT:-12:00:00}"
#SBATCH -o "${JOB_NAME:-nnunet_train}_%A_%a.out"
#SBATCH -e "${JOB_NAME:-nnunet_train}_%A_%a.err"
#SBATCH --array=${FOLDS_ARRAY:-0-4%4}

set -euo pipefail

# --- Environment/modules ---
module purge
module load GCC/${GCC_VERSION:-12.3.0} OpenMPI/${OPENMPI_VERSION:-4.1.5} Python/${PYTHON_VERSION:-3.11.3}

# Activate venv
VENV_PATH="${NNUNET_ENV:-$HOME/envs/nnunet}"
source "$VENV_PATH/bin/activate"

# --- nnU-Net paths ---
export nnUNet_raw="${NNUNET_RAW:-$PWD/nnUNet_raw}"
export nnUNet_preprocessed="${NNUNET_PREPROCESSED:-$PWD/nnUNet_preprocessed}"
export nnUNet_results="${NNUNET_RESULTS:-$PWD/nnUNet_results}"

# --- Params ---
DATASET="${DATASET_ID:-502}"
CFG="${NNUNET_CONFIG:-3d_fullres}"
TRAINER="${NNUNET_TRAINER:-nnUNetTrainer}"
PLANS="${NNUNET_PLANS:-nnUNetPlans}"

# Fold from SLURM array (fallback to 0 if run standalone)
FOLD="${SLURM_ARRAY_TASK_ID:-0}"

# Results dir per fold
OUT_DIR="$nnUNet_results/Dataset${DATASET}_CAC/${TRAINER}__${PLANS}__${CFG}/fold_${FOLD}"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

# Resume if checkpoint exists
RESUME_ARG=""
if [[ -f "$OUT_DIR/checkpoint_latest.pth" ]]; then
  echo "[INFO] Resuming from $OUT_DIR/checkpoint_latest.pth"
  RESUME_ARG="-c $OUT_DIR/checkpoint_latest.pth"
else
  echo "[INFO] No checkpoint detected"
fi

echo "[INFO] Host=$(hostname)"
echo "[INFO] Dataset=$DATASET | Config=$CFG | Fold=$FOLD"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] RAW=$nnUNet_raw | PREP=$nnUNet_preprocessed | RES=$nnUNet_results"

# --- Training ---
nnUNetv2_train "$DATASET" "$CFG" "$FOLD" -tr "$TRAINER" -p "$PLANS" $RESUME_ARG
