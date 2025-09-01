#!/usr/bin/env python3
"""
build_nnunet_dataset.py

This script prepares a dataset in the nnU-Net "raw" format from paired CT volumes
and their corresponding segmentation labelmaps.

Functionality:
- Collects CT images (*.nii.gz) from the specified input directory.
- Matches them with segmentation masks (*.nii.gz, *_seg suffix).
- Copies the files into an nnU-Net raw dataset structure:
    Dataset<ID>_<NAME>/{imagesTr, labelsTr}/
- Generates a dataset.json file describing:
    * Dataset metadata (name, description, licence, reference).
    * Modality mapping.
    * Label definitions.
    * Training case list (image + label paths).

Expected file naming:
    CTs:    <PatientID>_SER<NNN>_... .nii.gz
    Labels: <PatientID>_SER<NNN>_seg.nii.gz

Example usage:
    python build_nnunet_dataset.py \
        --images_dir /path/to/CTs \
        --labels_dir /path/to/labels \
        --out_root $nnUNet_raw \
        --ds_id 501 \
        --ds_name CAC_STD \
        --desc "Coronary vessels & mediastinum segmentation (non-contrast CT)."
"""

import os, re, json, shutil, glob, argparse


def main():
    """
    Main entry point of the script.
    Builds an nnU-Net raw dataset structure from paired CT images and segmentation labelmaps.

    The script performs the following steps:
    1. Collects CT volumes and their corresponding segmentation masks.
    2. Validates file naming and matching between images and labels.
    3. Copies them into the nnU-Net raw dataset structure:
       Dataset<ID>_<NAME>/{imagesTr, labelsTr}/
    4. Generates the dataset.json file with dataset metadata, label mapping, and training cases.

    Command-line arguments:
        --images_dir : Path to directory with CT NIfTI files (*.nii.gz).
        --labels_dir : Path to directory with segmentation masks (*.nii.gz, *_seg suffix).
        --out_root   : Root folder where the dataset directory will be created.
        --ds_id      : Numeric dataset ID (e.g., 501).
        --ds_name    : Short dataset name (e.g., CAC_STD).
        --desc       : Optional free-text dataset description (default provided).
    """
    ap = argparse.ArgumentParser(description="Build nnU-Net raw dataset from paired CT+labels")
    ap.add_argument("--images_dir", required=True, help="Folder with CT NIfTI files (.nii.gz)")
    ap.add_argument("--labels_dir", required=True, help="Folder with label NIfTI files (*_seg.nii.gz)")
    ap.add_argument("--out_root", required=True, help="Where to create nnUNet_raw/Dataset<ID>_<NAME>")
    ap.add_argument("--ds_id", required=True, help="Numeric dataset id, e.g. 501")
    ap.add_argument("--ds_name", required=True, help="Dataset short name, e.g. CAC_STD")
    ap.add_argument("--desc", default="Coronary vessels & mediastinum segmentation (non-contrast CT).",
                    help="Free-text description for dataset.json")
    args = ap.parse_args()

    ds_dir_name = f"Dataset{args.ds_id}_{args.ds_name}"
    out_ds = os.path.join(args.out_root, ds_dir_name)
    imagesTr = os.path.join(out_ds, "imagesTr")
    labelsTr = os.path.join(out_ds, "labelsTr")
    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)

    # Collect all CT image files
    cts = sorted(glob.glob(os.path.join(args.images_dir, "*.nii.gz")))
    cases = []

    def parse_pid_ser(fname):
        """
        Extract patient ID and series number from filename.

        Expected format: <PatientID>_SER<NNN>_... .nii.gz
        Example: 14JC2_SER002_STANDARD_1.25mm.nii.gz -> (14JC2, 002)

        Args:
            fname (str): Path to CT file.

        Returns:
            tuple: (PatientID, SeriesNumber) if valid, otherwise (None, None).
        """
        m = re.search(r"^([A-Za-z0-9]+)_SER(\d+)", os.path.basename(fname))
        return (m.group(1), m.group(2)) if m else (None, None)

    copied = 0
    for ct in cts:
        pid, ser = parse_pid_ser(ct)
        if not pid:
            print(f"[WARN] Cannot parse PatientID/Series from: {os.path.basename(ct)}")
            continue
        # Expected label name corresponding to the CT
        lab_src = os.path.join(args.labels_dir, f"{pid}_SER{ser}_seg.nii.gz")
        if not os.path.exists(lab_src):
            print(f"[WARN] Missing label for {pid}_SER{ser} (expected {os.path.basename(lab_src)})")
            continue

        # nnU-Net naming convention
        case_id = f"CAC_{pid}_SER{ser}"
        img_tgt = os.path.join(imagesTr, f"{case_id}_0000.nii.gz")
        lab_tgt = os.path.join(labelsTr, f"{case_id}.nii.gz")

        shutil.copy2(ct, img_tgt)
        shutil.copy2(lab_src, lab_tgt)
        cases.append(case_id)
        copied += 1
        print(f"[OK] Added case: {case_id}")

    # Create dataset.json describing dataset metadata and cases
    dataset_json = {
        "name": args.ds_name,
        "description": args.desc,
        "reference": "MSc thesis (2025)",
        "licence": "research-only",
        "release": "1.0",
        "tensorImageSize": "3D",
        "modality": {"0": "CT"},
        "labels": {
            "background": 0,
            "1": "mediastinum_coronary_field",
            "2": "LM_LAD",
            "3": "LCx",
            "4": "RCA"
        },
        "numTraining": len(cases),
        "numTest": 0,
        "training": [
            {"image": f"./imagesTr/{cid}_0000.nii.gz",
             "label": f"./labelsTr/{cid}.nii.gz"} for cid in cases
        ],
        "test": []
    }
    with open(os.path.join(out_ds, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\n[SUMMARY] Copied {copied} cases into {out_ds}")
    print("[OK] Wrote dataset.json")


if __name__ == "__main__":
    main()
