# language: python
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import cv2

# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

TEST_ROOT = PROJECT_ROOT / "prawn_2025_circ_small_v1"
MODELS_DIR = SCRIPT_DIR / "models_csv"

# Fallback if no models found in subfolder
IN_CSV_FALLBACK = MODELS_DIR / "baseline_linreg_eyes_from3kpts_test.csv"

SEED = 42
MAX_INSTANCES_PER_IMAGE = 20
IMG_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def label_to_image_path(test_root: Path, label_path: Path) -> Path:
    labels_dir = test_root / "labels"
    images_dir = test_root / "images"

    # Try to make relative to labels dir if it's a full path
    try:
        rel = label_path.relative_to(labels_dir)
    except ValueError:
        # If not relative to labels_dir, assume it's already a relative path or filename
        rel = label_path

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


def visualize_model(csv_path: Path) -> None:
    if not csv_path.exists():
        return

    print(f"Processing model: {csv_path.name}")

    # 1. Setup Output Directory
    # Create folder of visualization for that specific model
    out_dir = csv_path.parent / f"viz_{csv_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Read Data
    df = pd.read_csv(csv_path)

    # Ensure numeric columns
    cols = ["e_x", "e_y", "e_pred_x", "e_pred_y"]
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        print(f"  [Skip] Missing columns {missing_cols} in {csv_path.name}")
        return

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Calculate dist if missing (Evaluation.py computes it but doesn't save to input CSV)
    if "dist" not in df.columns:
        dx = df["e_x"] - df["e_pred_x"]
        dy = df["e_y"] - df["e_pred_y"]
        df["dist"] = np.sqrt(dx**2 + dy**2)

    # 3. Determine which images to visualize
    # "using the evaluation summary in that model folder"
    summary_csv = csv_path.parent / f"evaluation_summary_{csv_path.stem}.csv"
    chosen_labels = []

    if summary_csv.exists():
        try:
            summ_df = pd.read_csv(summary_csv)
            # Look for worst images identified in evaluation
            worst = summ_df[summ_df["group"] == "worst_images_meanDist"]
            if not worst.empty:
                # Take top 5 worst
                chosen_labels = worst["label_file"].astype(str).tolist()[:5]
                print(f"  Using {len(chosen_labels)} worst images from summary.")
        except Exception as e:
            print(f"  Could not read summary {summary_csv.name}: {e}")

    # Fallback to random if no summary or no worst images found
    if not chosen_labels:
        if "label_file" not in df.columns:
            print("  [Skip] No label_file column.")
            return

        unique_labels = df["label_file"].dropna().unique().tolist()
        if len(unique_labels) > 0:
            rng = np.random.default_rng(SEED)
            # Pick up to 2 random images
            count = min(len(unique_labels), 2)
            chosen_labels = rng.choice(unique_labels, size=count, replace=False).tolist()
            print(f"  Using {len(chosen_labels)} random images (no summary found).")
        else:
            print("  No labels found in CSV.")
            return

    # 4. Generate Visualizations
    for idx, lf_str in enumerate(chosen_labels, start=1):
        lf = Path(lf_str)

        try:
            img_path = label_to_image_path(TEST_ROOT, lf)
        except FileNotFoundError as e:
            print(f"  [Warn] {e}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [Warn] Could not read image: {img_path}")
            continue

        h, w = img.shape[:2]

        # Get predictions for this image
        sub = df[df["label_file"] == lf_str].copy()
        # Sort by dist descending (worst predictions first)
        sub = sub.sort_values("dist", ascending=False).head(MAX_INSTANCES_PER_IMAGE)

        for _, r in sub.iterrows():
            if pd.isna(r["e_x"]) or pd.isna(r["e_pred_x"]):
                continue

            gt_x = int(round(float(r["e_x"]) * w))
            gt_y = int(round(float(r["e_y"]) * h))
            pr_x = int(round(float(r["e_pred_x"]) * w))
            pr_y = int(round(float(r["e_pred_y"]) * h))

            cv2.line(img, (gt_x, gt_y), (pr_x, pr_y), (255, 255, 255), 2, cv2.LINE_AA)
            draw_point(img, gt_x, gt_y, (0, 255, 0), "GT")
            draw_point(img, pr_x, pr_y, (0, 0, 255), "Pred")

        mean_dist = float(sub["dist"].mean()) if len(sub) else 0.0

        # Title on image
        info_txt = f"{csv_path.stem[:15]}.. | {img_path.name} | n={len(sub)} | meanDist={mean_dist:.4f}"
        cv2.putText(
            img,
            info_txt,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        out_path = out_dir / f"viz_{idx}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), img)

    print(f"  Saved visualizations to: {out_dir}")


def main() -> None:
    # Look for model CSVs under MODELS_DIR / "models"
    models_root = MODELS_DIR / "models"
    csv_list = []
    if models_root.exists():
        csv_list = sorted(models_root.rglob("*.csv"))

    # If none found, fall back to IN_CSV_FALLBACK
    if not csv_list:
        if IN_CSV_FALLBACK.exists():
            csv_list = [IN_CSV_FALLBACK]
        else:
            print(f"No model CSVs found in {models_root} and fallback {IN_CSV_FALLBACK} missing.")
            return

    for csv_path in csv_list:
        # Skip summary files if they happen to be in the list
        if "evaluation_summary" in csv_path.name:
            continue

        try:
            visualize_model(csv_path)
        except Exception as e:
            print(f"Failed to visualize {csv_path}: {e}")


if __name__ == "__main__":
    main()