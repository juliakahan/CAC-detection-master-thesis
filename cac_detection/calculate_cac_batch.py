import os, sys, glob, csv, argparse

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

LABELS = {"mediastinum": 1, "lm_lad": 2, "lcx": 3, "rca": 4}


def read(path):
    """
    Read a NIfTI image from disk and return both the SimpleITK image and a NumPy array.

    Args:
        path (str): Path to the NIfTI file.

    Returns:
        Tuple[sitk.Image, np.ndarray]: The loaded image object and its voxel array (z, y, x).
    """
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    return img, arr


def resample_like(src_img, ref_img, is_label=False):
    """
    Resample a source image to match the geometry of a reference image.

    Uses nearest-neighbor interpolation for labelmaps and linear interpolation for intensity images.
    If geometry is already identical, returns the source image unchanged.

    Args:
        src_img (sitk.Image): Source image to resample.
        ref_img (sitk.Image): Reference image providing size/spacing/origin/direction.
        is_label (bool): If True, use nearest-neighbor interpolation.

    Returns:
        sitk.Image: Resampled image aligned to reference geometry.
    """
    same = (
        src_img.GetSize() == ref_img.GetSize()
        and np.allclose(src_img.GetSpacing(),   ref_img.GetSpacing())
        and np.allclose(src_img.GetOrigin(),    ref_img.GetOrigin())
        and np.allclose(src_img.GetDirection(), ref_img.GetDirection())
    )
    if same:
        return src_img
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    return sitk.Resample(src_img, ref_img, sitk.Transform(), interp, 0.0, src_img.GetPixelID())


def save_like(ref_img, arr, out_path):
    """
    Save a NumPy array as a NIfTI image, copying geometry from a reference image.

    Args:
        ref_img (sitk.Image): Image whose geometry (spacing, origin, direction) will be copied.
        arr (np.ndarray): Array data to save (z, y, x).
        out_path (str): Output file path (.nii or .nii.gz).
    """
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(ref_img)
    sitk.WriteImage(out, out_path)


def mm_per_voxel(img):
    """
    Compute voxel size components and voxel volume in mm^3.

    Args:
        img (sitk.Image): A SimpleITK image.

    Returns:
        Tuple[float, float, float, float]: (sx, sy, sz, voxel_volume_mm3)
    """
    sx, sy, sz = img.GetSpacing()
    return sx, sy, sz, (sx * sy * sz)


def agatston_for_mask(ct, roi_mask, hu_thr=130.0, min_area_mm2=1.0):
    """
    Compute Agatston-like score within a 3D ROI mask.

    Lesions are defined with 3D 26-connectivity from voxels meeting HU threshold and mask.
    For each connected component, the in-slice maximum area is multiplied by a density weight:
        weight = 1 for maxHU in [thr, 200), 2 for [200,300), 3 for [300,400), 4 for >=400.

    Args:
        ct (sitk.Image): CT image (intensity in HU).
        roi_mask (np.ndarray): Binary mask (same shape as CT array). If None, thresholding is global.
        hu_thr (float): HU threshold (default 130).
        min_area_mm2 (float): Minimum in-slice area (mm^2) for a lesion to count.

    Returns:
        Tuple[float, float, int, float]: (agatston_score, volume_mm3, n_lesions, max_HU)
    """
    ct_arr = sitk.GetArrayFromImage(ct).astype(np.float32)
    vox_hu = ct_arr >= float(hu_thr)
    vox = vox_hu & (roi_mask > 0) if roi_mask is not None else vox_hu
    if not np.any(vox):
        return 0.0, 0.0, 0, 0.0

    lab, n = ndi.label(vox, structure=np.ones((3, 3, 3), dtype=np.uint8))
    sx, sy, sz, vox_mm3 = mm_per_voxel(ct)
    px_area_mm2 = sx * sy
    min_pix_per_slice = max(1, int(np.ceil(min_area_mm2 / px_area_mm2)))

    agat = 0.0
    vol_mm3 = 0.0
    max_hu_global = 0.0

    for comp_id in range(1, n + 1):
        comp = (lab == comp_id)
        if not np.any(comp):
            continue

        comp_vox = int(comp.sum())
        vol_mm3 += comp_vox * vox_mm3

        comp_max = float(ct_arr[comp].max())
        max_hu_global = max(max_hu_global, comp_max)
        if comp_max >= 400:   w = 4
        elif comp_max >= 300: w = 3
        elif comp_max >= 200: w = 2
        elif comp_max >= hu_thr: w = 1
        else: w = 0

        # For Agatston, use the maximum in-slice area of the component
        per_slice = comp.reshape(comp.shape[0], -1).sum(axis=1)
        if per_slice.max() < min_pix_per_slice or w == 0:
            continue

        agat += float(per_slice.max() * px_area_mm2 * w)

    return float(agat), float(vol_mm3), int(n), float(max_hu_global)


def main(
    ct_dir,
    lab_dir,
    out_csv,
    out_masks_dir=None,
    hu_thr=130.0,
    min_area_mm2=1.0,
    save_vessels_any=False,
    mediastinum_dilate=0,
    strict_clip=False,
):
    """
    Batch-calculates Agatston-like CAC metrics for CT volumes using vessel/mediastinum masks.

    Expected inputs:
        - CT volumes named *'_0000.nii.gz' in ct_dir (nnU-Net convention for single-channel CT).
        - Corresponding labelmaps in lab_dir named '<case>.nii.gz', with labels:
              1: mediastinum_coronary_field, 2: LM+LAD, 3: LCx, 4: RCA.

    For each case:
        1) Optionally dilate mediastinum mask (in voxels).
        2) Clip artery masks by mediastinum (intersection). If clipping removes all HU>=thr candidates,
           optionally fall back to vessels-only masks unless --strict_clip is set.
        3) Compute per-region stats and Agatston-like scores.
        4) Save an optional debug mask set if requested.

    Args:
        ct_dir (str): Directory with CT inputs (*_0000.nii.gz).
        lab_dir (str): Directory with labelmaps (<case>.nii.gz).
        out_csv (str): Output CSV file path.
        out_masks_dir (Optional[str]): Where to save optional debug masks.
        hu_thr (float): HU threshold for calcium candidate voxels.
        min_area_mm2 (float): Minimum in-slice area (mm^2) for a lesion to be counted.
        save_vessels_any (bool): If True, save union masks for vessels (any / in mediastinum).
        mediastinum_dilate (int): Optional dilation radius (voxels) for mediastinum before clipping.
        strict_clip (bool): If True, never fall back to vessels-only when clipping removes all candidates.
    """
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if out_masks_dir:
        os.makedirs(out_masks_dir, exist_ok=True)

    rows = []
    ct_paths = sorted(glob.glob(os.path.join(ct_dir, "*_0000.nii.gz")))
    assert ct_paths, f"No CT files found in {ct_dir}"

    for ct_0000 in ct_paths:
        pid = os.path.basename(ct_0000).replace("_0000.nii.gz", "")
        ct_img, _ = read(ct_0000)

        # Expected label filename alongside
        lab_path = os.path.join(lab_dir, f"{pid}.nii.gz")
        if not os.path.exists(lab_path):
            print(f"[WARN] missing label for {pid}, skipping")
            continue

        lab_img, _ = read(lab_path)
        lab_img_rs = resample_like(lab_img, ct_img, is_label=True)
        lab = sitk.GetArrayFromImage(lab_img_rs).astype(np.int16)

        ct_arr = sitk.GetArrayFromImage(ct_img).astype(np.float32)
        hu_min, hu_max = float(ct_arr.min()), float(ct_arr.max())

        # --- binary masks per class ---
        medi_mask = (lab == LABELS["mediastinum"]).astype(np.uint8)
        lad_mask = (lab == LABELS["lm_lad"]).astype(np.uint8)
        lcx_mask = (lab == LABELS["lcx"]).astype(np.uint8)
        rca_mask = (lab == LABELS["rca"]).astype(np.uint8)

        # Dilation of mediastinum — important if mediastinum is tight around the vessels in a single-label setup
        if mediastinum_dilate > 0:
            rad = (int(mediastinum_dilate),) * 3
            medi_mask = sitk.GetArrayFromImage(
                sitk.BinaryDilate(sitk.GetImageFromArray(medi_mask), rad)
            ).astype(np.uint8)

        # Clip vessels to mediastinum region
        lad_in_medi = (lad_mask & medi_mask).astype(np.uint8)
        lcx_in_medi = (lcx_mask & medi_mask).astype(np.uint8)
        rca_in_medi = (rca_mask & medi_mask).astype(np.uint8)

        # Diagnostics: count voxels and HU>=thr before/after clipping
        thr_mask = (ct_arr >= hu_thr).astype(np.uint8)

        def cnt(mask):
            return int(mask.sum()), int((thr_mask & (mask > 0)).sum())

        vox_medi, ge_medi = cnt(medi_mask)
        vox_lad, ge_lad = cnt(lad_mask)
        vox_lcx, ge_lcx = cnt(lcx_mask)
        vox_rca, ge_rca = cnt(rca_mask)

        vox_ladM, ge_ladM = cnt(lad_in_medi)
        vox_lcxM, ge_lcxM = cnt(lcx_in_medi)
        vox_rcaM, ge_rcaM = cnt(rca_in_medi)

        print(f"[DBG] mediastinum vox={vox_medi}, HU>={hu_thr}={ge_medi}")
        print(f"[DBG] LAD vox={vox_lad} HU>={hu_thr}={ge_lad}  | clipped: vox={vox_ladM} HU>={hu_thr}={ge_ladM}")
        print(f"[DBG] LCx vox={vox_lcx} HU>={hu_thr}={ge_lcx}  | clipped: vox={vox_lcxM} HU>={hu_thr}={ge_lcxM}")
        print(f"[DBG] RCA vox={vox_rca} HU>={hu_thr}={ge_rca}  | clipped: vox={vox_rcaM} HU>={hu_thr}={ge_rcaM}")

        # If clipping removes all HU>=thr but vessels-only had candidates, fall back unless strict clipping is requested
        def pick_mask(orig, clipped, orig_ge, clipped_ge):
            if clipped_ge == 0 and orig_ge > 0 and not strict_clip:
                print("[WARN] Clipping to mediastinum removed all HU>=thr, falling back to vessels-only for this artery.")
                return orig
            return clipped

        lad_final = pick_mask(lad_mask, lad_in_medi, ge_lad, ge_ladM)
        lcx_final = pick_mask(lcx_mask, lcx_in_medi, ge_lcx, ge_lcxM)
        rca_final = pick_mask(rca_mask, rca_in_medi, ge_rca, ge_rcaM)

        # Optional debug masks
        if save_vessels_any and out_masks_dir:
            vessels_any = (lad_mask | lcx_mask | rca_mask).astype(np.uint8)
            vessels_in_mediastinum = (lad_in_medi | lcx_in_medi | rca_in_medi).astype(np.uint8)
            save_like(ct_img, vessels_any, os.path.join(out_masks_dir, f"{pid}_vessels_any.nii.gz"))
            save_like(ct_img, vessels_in_mediastinum,
                      os.path.join(out_masks_dir, f"{pid}_vessels_in_mediastinum.nii.gz"))

        def stat_for_mask(mask_uint8):
            """
            Compute summary statistics and Agatston-like metrics for a given mask.
            Returns a tuple:
                (vox_total, vox_ge_thr, vol_mm3, agatston, n_lesions, maxHU)
            """
            roi = mask_uint8
            vox_total = int(roi.sum())
            vox_ge_thr = int(((ct_arr >= hu_thr) & (roi > 0)).sum())
            agat, vol_mm3, n_les, max_hu = agatston_for_mask(ct_img, roi, hu_thr=hu_thr, min_area_mm2=min_area_mm2)
            return vox_total, vox_ge_thr, vol_mm3, agat, n_les, max_hu

        # Report mediastinum as-is
        mediastinum = stat_for_mask(medi_mask)
        # Arteries with clipping and potential fallback
        lm_lad = stat_for_mask(lad_final)
        lcx = stat_for_mask(lcx_final)
        rca = stat_for_mask(rca_final)

        agat_total = lm_lad[3] + lcx[3] + rca[3]
        vol_total  = lm_lad[2] + lcx[2] + rca[2]

        rows.append({
            "case": pid,
            "HU_min": hu_min, "HU_max": hu_max,

            "vox_mediastinum": mediastinum[0], "vox_ge130_mediastinum": mediastinum[1],
            "vol_mm3_mediastinum": mediastinum[2], "agatston_mediastinum": mediastinum[3],
            "lesions_mediastinum": mediastinum[4], "maxHU_mediastinum": mediastinum[5],

            # arteries clipped to mediastinum (with fallback if needed)
            "vox_lm_lad": lm_lad[0], "vox_ge130_lm_lad": lm_lad[1],
            "vol_mm3_lm_lad": lm_lad[2], "agatston_lm_lad": lm_lad[3],
            "lesions_lm_lad": lm_lad[4], "maxHU_lm_lad": lm_lad[5],

            "vox_lcx": lcx[0], "vox_ge130_lcx": lcx[1],
            "vol_mm3_lcx": lcx[2], "agatston_lcx": lcx[3],
            "lesions_lcx": lcx[4], "maxHU_lcx": lcx[5],

            "vox_rca": rca[0], "vox_ge130_rca": rca[1],
            "vol_mm3_rca": rca[2], "agatston_rca": rca[3],
            "lesions_rca": rca[4], "maxHU_rca": rca[5],

            "agatston_score": agat_total, "volume_total": vol_total
        })

        print(f"[OK] {pid}  HU:[{hu_min:.1f},{hu_max:.1f}]  "
              f"LAD∩M:{lm_lad[1]}  LCx∩M:{lcx[1]}  RCA∩M:{rca[1]}  "
              f"Agatston total (clipped):{agat_total:.1f}")

    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            w.writeheader()
        w.writerows(rows)
    print(f"\nSaved table: {out_csv} (rows: {len(rows)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Batch CAC calculation from CT + labelmaps")
    ap.add_argument("--mediastinum_dilate", type=int, default=0,
                    help="Optional dilation (in voxels) of mediastinum mask before clipping.")
    ap.add_argument("--strict_clip", action="store_true",
                    help="If set, never fall back to vessels-only when clipping removes all HU>=thr.")

    ap.add_argument("ct_dir", help="Directory with CT volumes (*_0000.nii.gz).")
    ap.add_argument("labels_dir", help="Directory with labelmaps (<case>.nii.gz).")
    ap.add_argument("out_csv", help="Path to output CSV with per-case metrics.")
    ap.add_argument("--out_masks_dir", default=None,
                    help="Optional directory to save debug masks.")
    ap.add_argument("--hu_thr", type=float, default=130.0,
                    help="HU threshold for calcium candidate voxels (default: 130).")
    ap.add_argument("--min_area_mm2", type=float, default=1.0,
                    help="Minimum in-slice area (mm^2) for a lesion to be counted (default: 1.0).")
    ap.add_argument("--save_vessels_any", action="store_true",
                    help="If set, save union-of-vessels masks (any / in mediastinum).")

    args = ap.parse_args()

    main(
        args.ct_dir,
        args.labels_dir,
        args.out_csv,
        out_masks_dir=args.out_masks_dir,
        hu_thr=args.hu_thr,
        min_area_mm2=args.min_area_mm2,
        save_vessels_any=args.save_vessels_any,
        mediastinum_dilate=args.mediastinum_dilate,
        strict_clip=args.strict_clip,
    )
