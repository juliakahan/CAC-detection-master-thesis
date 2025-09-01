import os
import csv
import numpy as np
import SimpleITK as sitk
import slicer
from scipy import ndimage as ndi

"""
CAC per-artery calculation utilities for 3D Slicer.

This module:
- Loads a CT volume and a labelmap (with labels: 1 mediastinum, 2 LM+LAD, 3 LCx, 4 RCA).
- Optionally dilates mediastinum and/or vessel masks.
- Computes Agatston-like scores per artery using 3D connected components:
  per-component max in-slice area [mm^2] × density weight (1..4) with HU threshold.
- Optionally clips vessel masks to the mediastinum region (with an optional fallback).
- Saves a per-artery calcium labelmap and a CSV summary, and visualizes results in Slicer.

Label codes expected in `segPath`:
    1: mediastinum_coronary_field
    2: LM + LAD
    3: LCx
    4: RCA
"""

LABELS = {"mediastinum": 1, "LM_LAD": 2, "LCx": 3, "RCA": 4}


def agatston_weight(h):
    """
    Map maximum HU to Agatston density weight.

    Args:
        h (float): Maximum Hounsfield Unit within a component.

    Returns:
        int: Weight in {0,1,2,3,4}.
    """
    return 0 if h < 130 else (1 if h < 200 else (2 if h < 300 else (3 if h < 400 else 4)))


def resampleLabelTo(ct, seg):
    """
    Resample a labelmap to match a CT volume geometry if needed.

    Uses nearest-neighbor interpolation and uint16 pixel type.

    Args:
        ct (sitk.Image): Reference CT image (geometry source).
        seg (sitk.Image): Label image to be aligned.

    Returns:
        sitk.Image: Resampled (or original) label image aligned to `ct`.
    """
    same = (
        ct.GetSize() == seg.GetSize()
        and np.allclose(ct.GetSpacing(), seg.GetSpacing())
        and np.allclose(ct.GetOrigin(), seg.GetOrigin())
        and np.allclose(ct.GetDirection(), seg.GetDirection())
    )
    if same:
        return seg
    return sitk.Resample(seg, ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)


def _agatston_for_binary_mask(ct_img, roi_mask_bool, hu_thr=130.0, min_area_mm2=1.0):
    """
    Compute Agatston-like metrics inside a boolean ROI.

    Pipeline: (HU>=thr ∧ ROI) → 3D connected components (26-neighborhood) →
              per-component max in-slice area × density weight.

    Args:
        ct_img (sitk.Image): CT image in HU.
        roi_mask_bool (np.ndarray[bool]): ROI mask array (z, y, x).
        hu_thr (float): HU threshold for candidate voxels (default: 130).
        min_area_mm2 (float): Minimal per-slice area for a lesion to count.

    Returns:
        Tuple[float, float, int, float]:
            (agatston_score, volume_mm3, n_components, global_max_HU)
    """
    if not np.any(roi_mask_bool):
        return 0.0, 0.0, 0, 0.0
    arrCT = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    vox = (arrCT >= float(hu_thr)) & roi_mask_bool
    if not np.any(vox):
        return 0.0, 0.0, 0, 0.0

    lab, n = ndi.label(vox, structure=np.ones((3, 3, 3), dtype=np.uint8))
    sx, sy, sz = ct_img.GetSpacing()
    px_area = sx * sy
    vox_vol = sx * sy * sz
    min_pix_per_slice = max(1, int(np.ceil(min_area_mm2 / px_area)))

    agat = 0.0
    vol = 0.0
    maxhu_global = 0.0
    for cid in range(1, n + 1):
        comp = (lab == cid)
        if not np.any(comp):
            continue
        vol += comp.sum() * vox_vol
        cmax = float(arrCT[comp].max())
        maxhu_global = max(maxhu_global, cmax)
        per_slice = comp.reshape(comp.shape[0], -1).sum(axis=1)
        if per_slice.max() < min_pix_per_slice:
            continue
        w = agatston_weight(cmax)
        agat += float(per_slice.max() * px_area * w)
    return float(agat), float(vol), int(n), float(maxhu_global)


def run_cac_in_slicer(
    ctPath,
    segPath,
    outDir,
    dilateVox=0,            # dilation for LM+LAD/LCx/RCA (in voxels)
    minAreaMM2=1.0,
    useMediastinum=False,   # enable/disable clipping to mediastinum
    mediastinumDilate=0,    # mediastinum dilation (in voxels)
    strictClip=False,       # if True: no fallback when clipping removes all candidates
    huThr=130.0,
    assignMode="per-artery" # "per-artery" (preferred) or "overlap" (legacy)
):
    """
    Compute per-artery CAC in 3D Slicer and visualize the results.

    Defaults match the batch pipeline: per-artery assignment, no dilation, vessels-only
    (unless `useMediastinum=True` to clip by mediastinum).

    Args:
        ctPath (str): Path to CT volume (.nii.gz).
        segPath (str): Path to labelmap with labels {1..4} (.nii.gz).
        outDir (str): Output directory (will be created if missing).
        dilateVox (int): Optional dilation (voxels) applied to artery masks.
        minAreaMM2 (float): Minimum per-slice area in mm^2 for lesion counting.
        useMediastinum (bool): If True, clip artery masks to mediastinum.
        mediastinumDilate (int): Optional dilation (voxels) of mediastinum before clipping.
        strictClip (bool): If True, never fall back to vessels-only when clipping removes all HU>=thr.
        huThr (float): HU threshold (default 130).
        assignMode (str): "per-artery" or "overlap" (legacy union→assign mode).

    Returns:
        Tuple[str, str]: Paths to (calcium_labelmap, csv_results).
    """
    os.makedirs(outDir, exist_ok=True)

    # 1) I/O and geometry
    ct = sitk.ReadImage(ctPath)
    seg = sitk.ReadImage(segPath)
    seg = resampleLabelTo(ct, seg)

    # 2) masks from labels
    medi = sitk.Equal(seg, LABELS["mediastinum"])
    lad  = sitk.Equal(seg, LABELS["LM_LAD"])
    lcx  = sitk.Equal(seg, LABELS["LCx"])
    rca  = sitk.Equal(seg, LABELS["RCA"])

    if dilateVox and dilateVox > 0:
        rad = (int(dilateVox),) * 3
        lad = sitk.BinaryDilate(lad, rad)
        lcx = sitk.BinaryDilate(lcx, rad)
        rca = sitk.BinaryDilate(rca, rad)
    if useMediastinum and mediastinumDilate and mediastinumDilate > 0:
        medi = sitk.BinaryDilate(medi, (int(mediastinumDilate),) * 3)

    # 3) to numpy + debug prints
    arrCT  = sitk.GetArrayFromImage(ct).astype(np.float32)
    arrMED = sitk.GetArrayFromImage(medi).astype(bool)
    arrLAD = sitk.GetArrayFromImage(lad).astype(bool)
    arrLCX = sitk.GetArrayFromImage(lcx).astype(bool)
    arrRCA = sitk.GetArrayFromImage(rca).astype(bool)

    print(f"[DBG] mediastinum vox={int(arrMED.sum())}, HU>={huThr}={int(((arrCT>=huThr)&arrMED).sum())}")
    for name, m in [("LAD", arrLAD), ("LCx", arrLCX), ("RCA", arrRCA)]:
        print(f"[DBG] {name} vox={int(m.sum())} HU>={huThr}={int(((arrCT>=huThr)&m).sum())}")

    def clip(mask_bool):
        return (mask_bool & arrMED) if (useMediastinum and np.any(arrMED)) else mask_bool

    # 4) computation
    perArt = {
        "LM+LAD": {"agat": 0.0, "vol": 0.0, "n": 0},
        "LCx":    {"agat": 0.0, "vol": 0.0, "n": 0},
        "RCA":    {"agat": 0.0, "vol": 0.0, "n": 0},
    }
    ca_arr = np.zeros_like(arrCT, dtype=np.uint8)

    if assignMode == "per-artery":
        for (name, base_mask, out_lab) in [
            ("LM+LAD", arrLAD, 1),
            ("LCx",    arrLCX, 2),
            ("RCA",    arrRCA, 3),
        ]:
            roi = clip(base_mask)
            if useMediastinum:
                ge_before = int(((arrCT >= huThr) & base_mask).sum())
                ge_after  = int(((arrCT >= huThr) & roi).sum())
                print(f"[DBG] {name}: HU>={huThr} before-clip={ge_before} | after-clip={ge_after}")
                if ge_before > 0 and ge_after == 0 and not strictClip:
                    print(f"[WARN] Clipping to mediastinum removed all HU>=thr for {name}, falling back to vessels-only.")
                    roi = base_mask

            agat, vol, nles, _ = _agatston_for_binary_mask(ct, roi, hu_thr=huThr, min_area_mm2=minAreaMM2)
            perArt[name]["agat"] += agat
            perArt[name]["vol"]  += vol
            perArt[name]["n"]    += nles
            ca_arr[(arrCT >= huThr) & roi] = out_lab

    else:
        # Legacy "union→assign" mode (kept for compatibility)
        thr = (arrCT >= huThr)
        vessels = arrLAD | arrLCX | arrRCA
        roi_union = (vessels & arrMED) if (useMediastinum and np.any(arrMED)) else vessels
        vox = thr & roi_union

        lab, n = ndi.label(vox, structure=np.ones((3, 3, 3), dtype=np.uint8))
        sx, sy, sz = ct.GetSpacing()
        px_area = sx * sy
        vox_vol = sx * sy * sz
        min_pix = max(1, int(np.ceil(minAreaMM2 / px_area)))

        for cid in range(1, n + 1):
            comp = (lab == cid)
            per_slice = comp.reshape(comp.shape[0], -1).sum(axis=1)
            if per_slice.max() < min_pix:
                continue
            ov = [
                ("LM+LAD", (comp & arrLAD).sum(), 1),
                ("LCx",    (comp & arrLCX).sum(), 2),
                ("RCA",    (comp & arrRCA).sum(), 3),
            ]
            name, ovn, out_lab = max(ov, key=lambda x: x[1])
            if ovn == 0:
                continue
            cmax = float(arrCT[comp].max())
            w = agatston_weight(cmax)
            perArt[name]["agat"] += float(per_slice.max() * px_area * w)
            perArt[name]["vol"]  += float(comp.sum()) * vox_vol
            perArt[name]["n"]    += 1
            ca_arr[comp] = out_lab

    # 5) outputs
    ca_img = sitk.GetImageFromArray(ca_arr.astype(np.uint8))
    ca_img.CopyInformation(ct)
    caPath = os.path.join(outDir, "calcium_per_artery.nii.gz")
    sitk.WriteImage(ca_img, caPath)

    csvPath = os.path.join(outDir, "cac_results.csv")
    with open(csvPath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["artery", "n_lesions", "agatston", "volume_mm3"])
        for k in ["LM+LAD", "LCx", "RCA"]:
            w.writerow([k, perArt[k]["n"], round(perArt[k]["agat"], 2), round(perArt[k]["vol"], 2)])
        w.writerow([
            "TOTAL",
            int(sum(v["n"] for v in perArt.values())),
            round(sum(v["agat"] for v in perArt.values()), 2),
            round(sum(v["vol"]  for v in perArt.values()), 2),
        ])

    # 6) visualize in Slicer
    ctNode = slicer.util.loadVolume(ctPath)
    caNode = slicer.util.loadLabelVolume(caPath)
    segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "CAC per-artery")
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(caNode, segNode)

    d = segNode.GetSegmentation()
    names  = {1: "LM+LAD-Ca", 2: "LCx-Ca", 3: "RCA-Ca"}
    colors = {1: (0.95, 0.35, 0.35), 2: (0.35, 0.95, 0.35), 3: (0.35, 0.50, 0.95)}
    for i in range(d.GetNumberOfSegments()):
        segId = d.GetNthSegmentID(i)
        labelVal = i + 1
        if labelVal in names:
            d.GetSegment(segId).SetName(names[labelVal])
        if labelVal in colors:
            d.GetSegment(segId).SetColor(*colors[labelVal])

    slicer.util.setSliceViewerLayers(background=ctNode, label=caNode)

    total_agat = sum(v["agat"] for v in perArt.values())
    print(f"[Slicer] Done. TOTAL Agatston (assignMode={assignMode}): {total_agat:.1f}")
    print("Outputs:", caPath, csvPath)
    return caPath, csvPath
