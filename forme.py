#!/usr/bin/env python3
"""
Convert YOLO-Pose labels (train + val) to Excel
in the same long-format as keypoints_from_label_txt.xlsx

Each row = one keypoint
"""

from pathlib import Path
import pandas as pd

# ============================================================
# PATHS
# ============================================================
DATASET_ROOT = Path(
    "/Users/taircarmon/Desktop/ml-course-project/prawn_2025_circ_small_v1"
)

LABELS_ROOT = DATASET_ROOT / "labels"

OUTPUT_XLSX = DATASET_ROOT / "keypoints_from_label_txt_test.xlsx"

# ============================================================
# CONFIG
# ============================================================
N_KEYPOINTS = 4    # carapace, eyes, rostrum, tail
VALUES_PER_KP = 3 # x, y, visibility

# ============================================================
# MAIN
# ============================================================
rows = []

for split in ["train", "val"]:
    labels_dir = LABELS_ROOT / split
    if not labels_dir.exists():
        print(f"Skipping missing split: {labels_dir}")
        continue

    label_files = sorted(labels_dir.glob("*.txt"))
    print(f"{split}: found {len(label_files)} label files")

    for label_path in label_files:
        image_stem = label_path.stem  # image name without extension

        with open(label_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        for obj_id, line in enumerate(lines):
            parts = [float(p) for p in line.split()]

            # YOLO-Pose format:
            # class cx cy w h kp1x kp1y kp1v kp2x kp2y kp2v ...
            kp_start = 5
            expected_len = kp_start + N_KEYPOINTS * VALUES_PER_KP

            if len(parts) < expected_len:
                raise ValueError(
                    f"{label_path.name}: expected ≥ {expected_len} values, got {len(parts)}"
                )

            for k in range(N_KEYPOINTS):
                x = parts[kp_start + 3 * k + 0]
                y = parts[kp_start + 3 * k + 1]
                v = parts[kp_start + 3 * k + 2]

                rows.append({
                    "image_stem": image_stem,
                    "object_id": obj_id,
                    "keypoint_index": k,
                    "x_norm": x,
                    "y_norm": y,
                    "visibility": int(v),
                    "split": split,   # NEW: train / val
                })

# ============================================================
# SAVE EXCEL
# ============================================================
df = pd.DataFrame(rows)

df.to_excel(OUTPUT_XLSX, index=False)

print("\nYOLO test dataset (train + val) converted successfully.")
print(f"Output Excel saved to:\n{OUTPUT_XLSX}")
print(f"Total rows (keypoints): {len(df)}")
print(f"Images with labels: {df['image_stem'].nunique()}")
print(f"Objects: {df[['image_stem','object_id']].drop_duplicates().shape[0]}")
print("\nSplit breakdown:")
print(df['split'].value_counts())
