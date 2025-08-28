#!/bin/bash
# Unpack dataset archive into nnU-Net raw structure

set -e

# Go to project root
cd "${DATASET_ROOT:-$PWD}"
mkdir -p "${nnUNet_raw}"

# Extract the dataset tarball into nnUNet_raw
tar -xzf Dataset501_CAC_STD.tgz -C "${nnUNet_raw}"

# Quick inspection of the structure
ls -R "${nnUNet_raw}/Dataset501_CAC_STD" | head -n 30

# Verify number of CT images vs. label maps
ls -1 "${nnUNet_raw}/Dataset501_CAC_STD/imagesTr" | wc -l
ls -1 "${nnUNet_raw}/Dataset501_CAC_STD/labelsTr" | wc -l
