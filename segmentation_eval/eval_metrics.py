#!/usr/bin/env python3
"""
Segmentation evaluation utilities and CLI.

Computes classical overlap metrics (Dice, IoU, Precision, Recall),
distance-based metrics (HD95, ASD), and optional Surface Dice at a
given tolerance, for multi-label NIfTI masks.

Usage example:
    python eval_masks.py \
        --ref  /path/to/reference_masks \
        --pred /path/to/prediction_masks \
        --labels 1 2 3 4 \
        --out /path/to/results/per_case.csv \
        --hd_tolerance 1.0
"""

import os
import glob
import argparse

import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from surface_distance import (
    compute_surface_distances,
    compute_surface_dice_at_tolerance,
)

# ------------------ classical metrics ------------------ #
def dice(tp: int, fp: int, fn: int, nan_on_empty: bool = False) -> float:
    """
    Compute the Dice Similarity Coefficient (DSC).

    Args:
        tp (int): Number of true positive voxels.
        fp (int): Number of false positive voxels.
        fn (int): Number of false negative voxels.
        nan_on_empty (bool, optional): If True, return NaN when both masks are empty.
            If False, return 1.0 in that case. Default is False.

    Returns:
        float: Dice score in [0, 1], or NaN if nan_on_empty=True and masks are empty.
    """
    denom = 2 * tp + fp + fn
    if denom > 0:
        return 2 * tp / denom
    return np.nan if nan_on_empty else 1.0


def jaccard(tp: int, fp: int, fn: int, nan_on_empty: bool = False) -> float:
    """
    Compute the Jaccard index (Intersection over Union, IoU).

    Args:
        tp (int): Number of true positive voxels.
        fp (int): Number of false positive voxels.
        fn (int): Number of false negative voxels.
        nan_on_empty (bool, optional): If True, return NaN when both masks are empty.
            If False, return 1.0 in that case. Default is False.

    Returns:
        float: Jaccard index in [0, 1], or NaN if nan_on_empty=True and masks are empty.
    """
    denom = tp + fp + fn
    if denom > 0:
        return tp / denom
    return np.nan if nan_on_empty else 1.0


def precision(tp: int, fp: int, nan_on_empty: bool = False) -> float:
    """
    Compute precision (positive predictive value).

    Args:
        tp (int): Number of true positive voxels.
        fp (int): Number of false positive voxels.
        nan_on_empty (bool, optional): If True, return NaN when denominator is zero.
            If False, return 1.0 in that case. Default is False.

    Returns:
        float: Precision in [0, 1], or NaN if nan_on_empty=True and denominator=0.
    """
    denom = tp + fp
    if denom > 0:
        return tp / denom
    return np.nan if nan_on_empty else 1.0


def recall(tp: int, fn: int, nan_on_empty: bool = False) -> float:
    """
    Compute recall (sensitivity).

    Args:
        tp (int): Number of true positive voxels.
        fn (int): Number of false negative voxels.
        nan_on_empty (bool, optional): If True, return NaN when denominator is zero.
            If False, return 1.0 in that case. Default is False.

    Returns:
        float: Recall in [0, 1], or NaN if nan_on_empty=True and denominator=0.
    """
    denom = tp + fn
    if denom > 0:
        return tp / denom
    return np.nan if nan_on_empty else 1.0


# ------------------ distance-based metrics ------------------ #
def hd95_asd(gt: np.ndarray, pr: np.ndarray, voxelspacing: tuple[float, float, float]) -> tuple[float, float]:
    """
    Compute the 95th percentile Hausdorff Distance (HD95) and the Average Surface Distance (ASD).

    Args:
        gt (ndarray): Ground truth binary mask.
        pr (ndarray): Predicted binary mask.
        voxelspacing (tuple of float): Voxel spacing in millimeters (x, y, z).

    Returns:
        tuple:
            - hd95 (float): 95th percentile Hausdorff Distance in millimeters.
            - asd (float): Symmetric Average Surface Distance in millimeters.

    Notes:
        - If both masks are empty, returns (0.0, 0.0).
        - If only one mask is empty, returns (inf, inf).
    """
    gt = gt.astype(bool)
    pr = pr.astype(bool)
    if gt.sum() == 0 and pr.sum() == 0:
        return 0.0, 0.0
    if gt.sum() == 0 or pr.sum() == 0:
        return float("inf"), float("inf")
    sd = compute_surface_distances(gt, pr, spacing_mm=voxelspacing)
    d_gt2pr = sd["distances_gt_to_pred"]
    d_pr2gt = sd["distances_pred_to_gt"]
    all_d = np.concatenate([d_gt2pr, d_pr2gt])
    hd95 = np.percentile(all_d, 95)
    asd = np.mean(all_d)
    return float(hd95), float(asd)


def surface_dice(
    gt: np.ndarray,
    pr: np.ndarray,
    voxelspacing: tuple[float, float, float],
    tol_mm: float,
) -> float:
    """
    Compute the Surface Dice coefficient at a given tolerance.

    Args:
        gt (ndarray): Ground truth binary mask.
        pr (ndarray): Predicted binary mask.
        voxelspacing (tuple of float): Voxel spacing in millimeters (x, y, z).
        tol_mm (float): Tolerance in millimeters for surface distance agreement.

    Returns:
        float: Surface Dice coefficient in [0, 1].

    Notes:
        - Returns 1.0 if both masks are empty.
        - Returns 0.0 if only one mask is empty.
    """
    gt = gt.astype(bool)
    pr = pr.astype(bool)
    if gt.sum() == 0 and pr.sum() == 0:
        return 1.0
    if gt.sum() == 0 or pr.sum() == 0:
        return 0.0
    sd = compute_surface_distances(gt, pr, spacing_mm=voxelspacing)
    return float(compute_surface_dice_at_tolerance(sd, tolerance_mm=tol_mm))


# ------------------ I/O helpers ------------------ #
def load_nifti(path: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    """
    Load a NIfTI file and return its voxel data and spacing.

    Args:
        path (str): Path to the NIfTI file (.nii or .nii.gz).

    Returns:
        tuple:
            - data (ndarray of int32): Image data array.
            - spacing (tuple of float): Voxel spacing (x, y, z) in millimeters.
    """
    img = nib.load(path)
    data = np.asarray(img.get_fdata(), dtype=np.int32)
    spacing = tuple(img.header.get_zooms()[:3])
    return data, spacing


# ------------------ main ------------------ #
def main():
    """
    Entry point for segmentation evaluation.

    - Parses command-line arguments.
    - Iterates over all reference cases and matches predictions by filename.
    - Computes metrics: Dice, IoU, Precision, Recall, HD95, ASD, and optionally Surface Dice.
    - Saves per-case metrics to CSV.
    - Aggregates per-label mean and standard deviation into a summary CSV.

    Command-line arguments:
        --ref (str): Directory with reference masks.
        --pred (str): Directory with prediction masks.
        --labels (list[int]): Labels to evaluate.
        --out (str): Output path for per-case CSV.
        --hd_tolerance (float, optional): Tolerance in mm for Surface Dice.
        --nan_on_empty (flag): Report NaN instead of 1.0 when masks are empty.

    Output:
        - <out>.csv: Per-case results.
        - <out>_summary.csv: Per-label mean, std, and case count.
    """
    ap = argparse.ArgumentParser(
        description="Segmentation evaluation: Dice/IoU/Precision/Recall + HD95/ASD (+ optional Surface Dice@tol)"
    )
    ap.add_argument("--ref", required=True, help="Directory with reference masks (.nii or .nii.gz)")
    ap.add_argument(
        "--pred",
        required=True,
        help="Directory with predicted masks (.nii or .nii.gz) with identical filenames",
    )
    ap.add_argument(
        "--labels", nargs="+", type=int, required=True, help="List of labels to evaluate, e.g. 1 2 3"
    )
    ap.add_argument("--out", required=True, help="Path to CSV with per-case results")
    ap.add_argument(
        "--hd_tolerance",
        type=float,
        default=None,
        help="If provided (mm), compute Surface Dice @ tolerance",
    )
    ap.add_argument(
        "--nan_on_empty",
        action="store_true",
        help="If set, metrics on empty masks are reported as NaN instead of 1.0",
    )
    args = ap.parse_args()

    # Find all reference files (expects identical basenames in predictions)
    ref_paths = sorted(glob.glob(os.path.join(args.ref, "*.nii*")))
    if len(ref_paths) == 0:
        raise SystemExit(f"No reference files found in {args.ref}")

    rows = []
    for ref_p in tqdm(ref_paths, desc="Evaluating cases", unit="case"):
        case = os.path.basename(ref_p)
        pred_p = os.path.join(args.pred, case)
        if not os.path.exists(pred_p):
            tqdm.write(f"[WARN] Missing prediction for {case}")
            continue

        try:
            ref, sp = load_nifti(ref_p)
            pred, _ = load_nifti(pred_p)
        except Exception as e:
            tqdm.write(f"[ERROR] Cannot load {case}: {e}")
            continue

        for lab in args.labels:
            # Build binary masks for current label
            gt = (ref == lab).astype(np.uint8)
            pr = (pred == lab).astype(np.uint8)

            # Confusion terms
            tp = int(np.logical_and(gt == 1, pr == 1).sum())
            fp = int(np.logical_and(gt == 0, pr == 1).sum())
            fn = int(np.logical_and(gt == 1, pr == 0).sum())

            # Overlap metrics
            dsc = dice(tp, fp, fn, args.nan_on_empty)
            iou = jaccard(tp, fp, fn, args.nan_on_empty)
            prec = precision(tp, fp, args.nan_on_empty)
            rec = recall(tp, fn, args.nan_on_empty)

            # Surface-based metrics
            hd95, asd = hd95_asd(gt, pr, sp)

            row = {
                "case": case,
                "label": lab,
                "dice": dsc,
                "iou": iou,
                "precision": prec,
                "recall": rec,
                "hd95_mm": hd95,
                "asd_mm": asd,
            }
            if args.hd_tolerance is not None:
                row[f"surf_dice@{args.hd_tolerance}mm"] = surface_dice(gt, pr, sp, args.hd_tolerance)

            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No matching REF/PRED pairs with results.")

    # Save per-case results
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    # Summary per label (mean/std + n)
    agg = {
        "dice": ["mean", "std"],
        "iou": ["mean", "std"],
        "precision": ["mean", "std"],
        "recall": ["mean", "std"],
        "hd95_mm": ["mean", "std"],
        "asd_mm": ["mean", "std"],
    }
    if args.hd_tolerance is not None:
        agg[f"surf_dice@{args.hd_tolerance}mm"] = ["mean", "std"]

    summary = df.groupby("label").agg(agg)
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns.values]
    n_per_label = df.groupby("label")["case"].count().rename("n").reset_index()
    summary = summary.reset_index().merge(n_per_label, on="label", how="left")

    summ_path = os.path.splitext(args.out)[0] + "_summary.csv"
    summary.to_csv(summ_path, index=False)

    print(f"Saved: {args.out}")
    print(f"Saved: {summ_path}")


if __name__ == "__main__":
    main()
