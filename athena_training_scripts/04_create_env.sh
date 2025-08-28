#!/bin/bash
# Create and activate Python virtual environment for nnU-Net

set -e

# Path to virtual environment (override with NNUNET_ENV)
VENV_PATH="${NNUNET_ENV:-$HOME/envs/nnunet}"

python3 -m venv --system-site-packages "$VENV_PATH"
source "$VENV_PATH/bin/activate"

# Check Python version
python -V

# Upgrade pip & core tools
pip install --upgrade pip wheel setuptools
