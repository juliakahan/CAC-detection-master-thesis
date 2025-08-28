#!/bin/bash
# Install PyTorch + CUDA and nnU-Net v2 with dependencies

set -e

# Activate environment
VENV_PATH="${NNUNET_ENV:-$HOME/envs/nnunet}"
source "$VENV_PATH/bin/activate"

# Install PyTorch with CUDA (default: 12.1 for A100 GPU)
TORCH_VERSION="${TORCH_VERSION:-2.3.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.18.1}"
CUDA_TAG="${CUDA_TAG:-cu121}"

pip install --no-cache-dir torch=="${TORCH_VERSION}+${CUDA_TAG}" \
    torchvision=="${TORCHVISION_VERSION}+${CUDA_TAG}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# Install medical imaging libraries
pip install --no-cache-dir "numpy<2" SimpleITK nibabel scikit-image tqdm batchgenerators itk

# Install nnU-Net v2
pip install --no-cache-dir nnunetv2
