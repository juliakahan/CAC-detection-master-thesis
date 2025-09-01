# CAC_work/scripts/check_labels.py
import os
import glob
import numpy as np
import SimpleITK as sitk

def main():
    """
    Check segmentation labelmaps for consistency of class labels.

    The script:
    1. Scans a given root directory for NIfTI labelmaps (*_seg.nii.gz).
    2. Loads each segmentation and extracts unique label values.
    3. Verifies that all labels belong to the expected set {0, 1, 2, 3, 4}.
    4. Prints a warning if unexpected labels are detected.

    Expected label mapping:
        0 - background
        1 - mediastinum_coronary_field
        2 - LM_LAD
        3 - LCx
        4 - RCA
    """

    # Root folder containing labelmaps (*.nii.gz)
    # Change this path to your dataset location
    root = "/path/to/CAC_work"

    # Collect all segmentation files
    paths = glob.glob(os.path.join(root, "*_seg.nii.gz"))
    ok = True

    for p in paths:
        arr = sitk.GetArrayFromImage(sitk.ReadImage(p))
        u = np.unique(arr)
        print(os.path.basename(p), "unique labels:", u)
        if not set(u).issubset({0, 1, 2, 3, 4}):
            ok = False

    print("[OK] Labels within {0,1,2,3,4}" if ok else "[WARN] Unexpected labels detected!")

if __name__ == "__main__":
    main()
