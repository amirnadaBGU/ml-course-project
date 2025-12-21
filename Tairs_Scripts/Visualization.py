#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import cv2

# ============================================================
# CONFIG (no terminal needed)
# File location: ml-course-project/Tairs_Scripts/Visualization.py
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

TEST_ROOT = PROJECT_ROOT / "prawn_2025_circ_small_v1"

# READ + SAVE HERE (same folder you requested)
MODELS_DIR = SCRIPT_DIR / "models_csv"
IN_CSV = MODELS_DIR / "baseline_linreg_eyes_from3kpts_test.csv"
OUT_DIR = MODELS_DIR / "Tairs_Scripts/models_csv/models/BaseLine" / "viz_baseline"

SEED = 42
PICK_TWO_RANDOM_IMAGES = True
MAX_INSTANCES_PER_IMAGE = 20

# Manual override (set PICK_TWO_RANDOM_IMAGES=False):
LABEL1 = None
LABEL2 = None

IMG_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def label_to_image_path(test_root: Path, label_path: Path) -> Path:
    labels_dir = test_root / "labels"
    images_dir = test_root / "images"

    rel = label_path.relative_to(labels_dir)
    stem_rel = rel.with_suffix("")

    for ext in IMG_EXTS:
        cand = images_dir / (str(stem_rel) + ext)
        if cand.exists():
            return cand

    for ext in IMG_EXTS:
        cand = images_dir / (stem_rel.name + ext)
        if cand.exists():
            return cand

    raise FileNotFoundError(f"Could not find matching image for label: {label_path}")


def draw_point(img, x_px: int, y_px: int, color, text: str) -> None:
    cv2.circle(img, (x_px, y_px), 6, color, -1)
    cv2.putText(img, text, (x_px + 8, y_px - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing CSV:\n{IN_CSV}\nRun Baseline.py first.")

    df = pd.read_csv(IN_CSV)
    needed = {"label_file", "e_x", "e_y", "e_pred_x", "e_pred_y", "dist"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    unique_labels = df["label_file"].dropna().unique().tolist()
    if len(unique_labels) < 2:
        raise ValueError("CSV must include at least 2 different label files.")

    if PICK_TWO_RANDOM_IMAGES:
        rng = np.random.default_rng(SEED)
        chosen_labels = rng.choice(unique_labels, size=2, replace=False).tolist()
    else:
        if LABEL1 is None or LABEL2 is None:
            raise ValueError("Set LABEL1 and LABEL2 or set PICK_TWO_RANDOM_IMAGES=True.")
        chosen_labels = [LABEL1, LABEL2]

    for idx, lf_str in enumerate(chosen_labels, start=1):
        lf = Path(lf_str)

        img_path = label_to_image_path(TEST_ROOT, lf)
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        h, w = img.shape[:2]

        sub = df[df["label_file"] == lf_str].copy()
        sub = sub.sort_values("dist", ascending=False).head(MAX_INSTANCES_PER_IMAGE)

        for _, r in sub.iterrows():
            gt_x = int(round(float(r["e_x"]) * w))
            gt_y = int(round(float(r["e_y"]) * h))
            pr_x = int(round(float(r["e_pred_x"]) * w))
            pr_y = int(round(float(r["e_pred_y"]) * h))

            cv2.line(img, (gt_x, gt_y), (pr_x, pr_y), (255, 255, 255), 2, cv2.LINE_AA)
            draw_point(img, gt_x, gt_y, (0, 255, 0), "Eyes GT")
            draw_point(img, pr_x, pr_y, (0, 0, 255), "Eyes Pred")

        mean_dist = float(sub["dist"].mean()) if len(sub) else float("nan")
        cv2.putText(
            img,
            f"Baseline mean-eyes | {img_path.name} | drawn={len(sub)} | meanDist={mean_dist:.4f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        out_path = OUT_DIR / f"viz_{idx}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), img)
        print(f"Saved: {out_path}")

    print(f"\nAll visualizations saved to:\n{OUT_DIR}")


if __name__ == "__main__":
    main()
