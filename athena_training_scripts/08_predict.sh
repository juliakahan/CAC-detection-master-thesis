#!/bin/bash
#SBATCH -J "${JOB_NAME:-pred_ens}"
#SBATCH -A "${ACCOUNT:-default-account}"
#SBATCH -p "${PARTITION:-gpu}"
#SBATCH --gres="gpu:${GPU_TYPE:-a100}:${NUM_GPUS:-1}"
#SBATCH --cpus-per-task="${CPUS_PER_TASK:-4}"
#SBATCH --mem="${MEMORY:-32G}"
#SBATCH -t "${TIME_LIMIT:-02:00:00}"
#SBATCH -o "${JOB_NAME:-pred_ens}_%j.out"
#SBATCH -e "${JOB_NAME:-pred_ens}_%j.err"

set -euo pipefail

# ===== Config =====
DATASET="${DATASET_ID:-502}"
CONFIG="${NNUNET_CONFIG:-3d_fullres}"
FOLDS="${NNUNET_FOLDS:-0 1 2 3 4}"

# Test-Time Augmentation (default ON in nnU-Net v2)
TTA="${TTA:-1}"

# Worker counts
NIO="${NIO:-2}"    # I/O workers
NPP="${NPP:-2}"    # preprocessing workers

# ===== Paths =====
export nnUNet_raw="${NNUNET_RAW:-$PWD/nnUNet_raw}"
export nnUNet_preprocessed="${NNUNET_PREPROCESSED:-$PWD/nnUNet_preprocessed}"
export nnUNet_results="${NNUNET_RESULTS:-$PWD/nnUNet_results}"

IN_DIR="$nnUNet_raw/Dataset${DATASET}_CAC/imagesTr"
OUT_DIR="$nnUNet_results/Dataset${DATASET}_CAC/preds_ensTr$( [ "$TTA" = "1" ] && echo "_tta" )"

# ===== Modules + env =====
module purge
module load GCC/${GCC_VERSION:-12.3.0} OpenMPI/${OPENMPI_VERSION:-4.1.5} Python/${PYTHON_VERSION:-3.11.3}

VENV_PATH="${NNUNET_ENV:-$HOME/envs/nnunet}"
source "$VENV_PATH/bin/activate"

echo "== Node: $(hostname)"
echo "== Dataset: ${DATASET}, Config: ${CONFIG}, Folds: ${FOLDS}, TTA: ${TTA}"
echo "== IN : $IN_DIR"
echo "== OUT: $OUT_DIR"
echo "== nnUNetv2_predict: $(which nnUNetv2_predict || echo 'NOT FOUND')"

# Sanity checks
[ -d "$IN_DIR" ] || { echo "[ERR] Missing input dir: $IN_DIR"; exit 2; }
mkdir -p "$OUT_DIR"

# TTA flag (disable with --disable_tta)
TTA_FLAG=()
if [ "$TTA" = "0" ]; then
  TTA_FLAG=(--disable_tta)
fi

CHK="checkpoint_best.pth"

set -x
nnUNetv2_predict \
  -i "$IN_DIR" \
  -o "$OUT_DIR" \
  -d "$DATASET" \
  -c "$CONFIG" \
  -f $FOLDS \
  -chk "$CHK" \
  "${TTA_FLAG[@]}" \
  --save_probabilities \
  -npp "$NPP" \
  -nps "$NIO"
set +x

echo "== Done. Example outputs:"
ls -lh "$OUT_DIR" | head -n 50
