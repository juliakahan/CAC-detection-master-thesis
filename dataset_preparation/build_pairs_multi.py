#!/usr/bin/env python3
"""
build_pairs_multi.py

Create a CSV manifest that maps CT volumes (*.nii.gz) to their corresponding
segmentation files (*.nrrd or *.seg.nrrd). The script infers PatientID from
filenames and prepares a target output path for the converted labelmap
(NIfTI, *_seg.nii.gz) per case.

Assumptions:
- CT filenames contain the pattern: <PatientID>_SER<NNN>_... .nii.gz
- Segmentation filenames are stored as: <PatientID>.nrrd or <PatientID>.seg.nrrd
  (i.e., they do NOT include the _SER<NNN> part; they are per-patient files)

Output CSV schema:
    patient_id, series_tag, ct_path, seg_path, out_labelmap

Example usage:
    python build_pairs_multi.py \
        --images_dir /path/CT_nii \
        --labels_dir /path/Slicer_segs_nrrd \
        --out_csv /path/manifests/pairs_multi.csv
"""

import os, glob, csv, argparse, re

# Parse command-line arguments
ap = argparse.ArgumentParser(description="Build a CSV manifest mapping CT images to segmentation files")
ap.add_argument("--images_dir", required=True, help="Folder containing CT NIfTI files (*.nii.gz)")
ap.add_argument("--labels_dir", required=True, help="Folder containing segmentation files (*.nrrd or *.seg.nrrd)")
ap.add_argument("--out_csv", required=True, help="Output CSV manifest path (e.g., pairs_multi.csv)")
args = ap.parse_args()

# Ensure the output directory exists
os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

def pid_from_ct(fn):
    """
    Extract the PatientID from a CT filename.

    Expected CT filename format:
        <PatientID>_SER<NNN>_... .nii.gz

    Args:
        fn (str): CT file path.

    Returns:
        str: PatientID parsed from the basename (substring before `_SER`).
    """
    return re.split(r"_SER", os.path.basename(fn))[0]

# Build an index mapping PatientID -> segmentation file path
seg_index = {}
for seg_path in glob.glob(os.path.join(args.labels_dir, "*.nrrd")):
    base = os.path.basename(seg_path)
    # Strip either ".seg.nrrd" or ".nrrd" to get PatientID
    pid = re.sub(r"\.seg\.nrrd$|\.nrrd$", "", base)
    seg_index[pid] = seg_path

rows = []
for ct_path in glob.glob(os.path.join(args.images_dir, "*.nii.gz")):
    pid = pid_from_ct(ct_path)
    seg_path = seg_index.get(pid)
    if not seg_path:
        print(f"[WARN] No segmentation for patient {pid}")
        continue

    # Extract series number tag from CT filename (e.g., _SER002 -> SER002)
    m = re.search(r"_SER(\d+)", os.path.basename(ct_path))
    ser_tag = f"SER{m.group(1)}" if m else "SERxxx"

    # Define where a converted labelmap (.nii.gz) would be written later
    out_labelmap = os.path.join(
        os.path.dirname(args.out_csv),
        f"{pid}_{ser_tag}_seg.nii.gz"
    )

    rows.append({
        "patient_id": pid,
        "series_tag": ser_tag,
        "ct_path": os.path.abspath(ct_path),
        "seg_path": os.path.abspath(seg_path),
        "out_labelmap": os.path.abspath(out_labelmap)
    })

# Write the CSV manifest
with open(args.out_csv, "w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=["patient_id", "series_tag", "ct_path", "seg_path", "out_labelmap"]
    )
    w.writeheader()
    w.writerows(rows)

print(f"[OK] wrote {args.out_csv} with {len(rows)} rows")
