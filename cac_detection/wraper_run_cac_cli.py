#!/usr/bin/env python3
"""
Thin CLI wrapper to run `run_cac_in_slicer()` from an external tools file.

Usage (from terminal, via Slicer):
    /Applications/Slicer.app/Contents/MacOS/Slicer \
      --python-script /path/to/run_cac_cli.py -- \
      --tools /path/to/run_cac_in_slicer.py \
      --ct /path/to/DatasetXXX/imagesTr/CAC_<PID>_SER<NNN>_0000.nii.gz \
      --seg /path/to/predictions/CAC_<PID>_SER<NNN>.nii.gz \
      --out /path/to/cac_results/cac_masks \
      --dilate 0 \
      --area 1.0 \
      --mediastinum \
      --mediastinum-dilate 5 \
      --hu 130 \
      --assign per-artery

Notes:
- The `--tools` argument must point to a Python file that defines a function:
      run_cac_in_slicer(ctPath, segPath, outDir, dilateVox, minAreaMM2,
                        useMediastinum, mediastinumDilate, strictClip=False,
                        huThr=130.0, assignMode="per-artery")
- This wrapper loads that file with `runpy.run_path()` and invokes the function.
"""

import sys
import argparse
import runpy
import os


def main(argv=None):
    """Parse args, import tools file, and call run_cac_in_slicer()."""
    p = argparse.ArgumentParser(description="Run per-artery CAC from Slicer via external tools file")
    p.add_argument("--tools", required=True,
                   help="Path to a .py file that defines run_cac_in_slicer()")
    p.add_argument("--ct",    required=True, help="Path to CT volume (.nii/.nii.gz)")
    p.add_argument("--seg",   required=True, help="Path to labelmap with labels {1..4} (.nii/.nii.gz)")
    p.add_argument("--out",   required=True, help="Output directory")

    p.add_argument("--dilate", type=int, default=0,
                   help="Vessel dilation in voxels (0=off)")
    p.add_argument("--area",   type=float, default=1.0,
                   help="Minimum per-slice area in mm^2")
    p.add_argument("--mediastinum", action="store_true",
                   help="If set, clip vessel masks to mediastinum")
    p.add_argument("--mediastinum-dilate", type=int, default=0,
                   help="Mediastinum dilation in voxels")
    p.add_argument("--hu",     type=float, default=130.0,
                   help="HU threshold for calcium candidates")
    p.add_argument("--assign", choices=["per-artery", "overlap"], default="per-artery",
                   help="Assignment mode: 'per-artery' (preferred) or 'overlap' (legacy)")
    # Optional: expose strict clipping if the tools function supports it
    p.add_argument("--strict-clip", action="store_true",
                   help="If set, no fallback to vessels-only when clipping removes all candidates")

    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    # Basic sanity checks
    if not os.path.isfile(args.tools):
        raise FileNotFoundError(f"--tools file not found: {args.tools}")
    if not os.path.isfile(args.ct):
        raise FileNotFoundError(f"--ct file not found: {args.ct}")
    if not os.path.isfile(args.seg):
        raise FileNotFoundError(f"--seg file not found: {args.seg}")
    os.makedirs(args.out, exist_ok=True)

    # Load the tools file that should define run_cac_in_slicer()
    ns = runpy.run_path(args.tools)
    if "run_cac_in_slicer" not in ns or not callable(ns["run_cac_in_slicer"]):
        raise KeyError(f"`run_cac_in_slicer` not found or not callable in: {args.tools}")
    run_cac = ns["run_cac_in_slicer"]

    print("[CLI] Launching CAC in Slicer GUI…")
    caPath, csvPath = run_cac(
        ctPath=args.ct,
        segPath=args.seg,
        outDir=args.out,
        dilateVox=args.dilate,
        minAreaMM2=args.area,
        useMediastinum=args.mediastinum,
        mediastinumDilate=args.mediastinum_dilate,
        huThr=args.hu,
        assignMode=args.assign,
        # Pass strict clip if the underlying function supports it; extra kwargs are ignored otherwise
        strictClip=getattr(args, "strict_clip", False),
    )
    print("[CLI] Done.")
    print("[CLI] Outputs:")
    print("  - calcium labelmap:", caPath)
    print("  - results CSV     :", csvPath)


if __name__ == "__main__":
    main()
