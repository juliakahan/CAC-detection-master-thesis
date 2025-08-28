#!/bin/bash
# Request an interactive compute node with GPU

srun -A "${ACCOUNT:-default-account}" \
     -N 1 --ntasks-per-node=1 \
     --cpus-per-task="${CPUS_PER_TASK:-8}" \
     --gres="gpu:${GPU_TYPE:-a100}:${NUM_GPUS:-1}" \
     -t "${TIME_LIMIT:-04:00:00}" \
     -p "${PARTITION:-gpu}" \
     --pty /bin/bash -l
