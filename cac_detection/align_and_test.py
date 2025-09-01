#!/usr/bin/env python3
"""
Resample a segmentation mask to match CT geometry (if needed) and
report basic ROI statistics, including HU range and voxel counts >= 130 HU
within selected coronary-related labels.

Usage:
    python script.py <ct_path> <label_path> <out_dir>

Args:
    ct_path (str): Path to CT volume (.nii or .nii.gz).
    label_path (str): Path to segmentation labelmap (.nii or .nii.gz).
    out_dir (str): Output directory for any auxiliary files.
"""

import SimpleITK as sitk
import numpy as np
import sys
import os


ct_path = sys.argv[1]            # e.g., /path/to/CT.nii.gz
label_path = sys.argv[2]         # e.g., /path/to/seg_mask.nii.gz
out_dir = sys.argv[3]            # directory for outputs

os.makedirs(out_dir, exist_ok=True)


def info(img):
    """
    Collect basic image geometry information.

    Returns:
        dict: size, spacing, origin, and direction (rounded) of the image.
    """
    return {
        "size": img.GetSize(),
        "spacing": img.GetSpacing(),
        "origin": img.GetOrigin(),
        "direction": tuple(np.round(list(img.GetDirection()), 6)),
    }


def resample_like(moving, reference, is_label=True):
    """
    Resample 'moving' image onto the grid of 'reference'.

    Args:
        moving (sitk.Image): Image to be resampled (typically the labelmap).
        reference (sitk.Image): Target geometry (typically the CT).
        is_label (bool): Use nearest-neighbor for labels, linear for images.

    Returns:
        sitk.Image: Resampled image.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(moving)


def sitk_to_np(img):
    """
    Convert a SimpleITK image to a NumPy array (z, y, x order).
    """
    return sitk.GetArrayFromImage(img)


def unique_counts(a):
    """
    Compute unique values and their counts in an array.

    Returns:
        dict: {value: count}
    """
    u, c = np.unique(a, return_counts=True)
    return dict(zip(u.tolist(), c.tolist()))


# 1) Read inputs (SimpleITK uses LPS; direction/origin preserve orientation)
ct = sitk.ReadImage(ct_path)
lab = sitk.ReadImage(label_path)

print("CT:", info(ct))
print("LAB:", info(lab))


def same_geometry(a, b):
    """
    Check if two images share identical size, spacing, origin, and direction.
    """
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing())
        and np.allclose(a.GetOrigin(), b.GetOrigin())
        and np.allclose(a.GetDirection(), b.GetDirection())
    )


# 2) If geometry differs → resample labelmap to CT grid
if not same_geometry(ct, lab):
    print(">>> Geometries differ — resampling labelmap to CT grid...")
    lab_rs = resample_like(lab, ct, is_label=True)
    sitk.WriteImage(lab_rs, os.path.join(out_dir, "label_resampled_to_CT.nii.gz"))
    lab = lab_rs
else:
    print(">>> Geometries match.")

# 3) Inspect present labels
lab_np = sitk_to_np(lab).astype(np.int32)
print("Unique labels and counts:", unique_counts(lab_np))

# 4) Compute global HU range and voxels >= 130 HU inside selected ROIs
ct_np = sitk_to_np(ct).astype(np.float32)

# DEFINE LABEL MAPPING — ADJUST TO YOUR MASK CONVENTION!
# Example (change values if different in your data):
mapping = {
    "mediastinum": 1,
    "lm_lad": 2,
    "lcx": 3,
    "rca": 4,
}


def count_roi(label_value):
    """
    Count voxels within a label and those with HU >= 130 inside that label.

    Returns:
        tuple: (voxels_in_roi, voxels_ge_130HU)
    """
    roi = (lab_np == label_value)
    vox_roi = int(roi.sum())
    vox_130 = int((ct_np[roi] >= 130).sum()) if vox_roi > 0 else 0
    return vox_roi, vox_130


# Global HU range
print(f"HU range (global): {ct_np.min()} {ct_np.max()}")

# Per-ROI counts
for name, lv in mapping.items():
    vox, vox130 = count_roi(lv)
    print(f"{name}: vox={vox} | vox HU>=130={vox130}")

# 5) Save a helper mask for quick QA in 3D Slicer:
#    a binary mask of any coronary vessel (LM+LAD, LCx, RCA)
vessels = np.isin(lab_np, [mapping["lm_lad"], mapping["lcx"], mapping["rca"]]).astype(np.uint8)
vessels_img = sitk.GetImageFromArray(vessels)
vessels_img.CopyInformation(ct)  # ensure geometry matches CT
sitk.WriteImage(vessels_img, os.path.join(out_dir, "vessels_any.nii.gz"))
print("Saved vessels_any.nii.gz (CT geometry).")
