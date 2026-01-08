from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import cv2

try:
    # Preferred when imported as a package
    from Tairs_Scripts_norm.config import TEST_XLSX, OUTPUT_BASE
except Exception:
    # Fallback when executed directly: add project root to sys.path and import
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[0].parents[0]
    # ROOT now points to Tairs_Scripts_norm/ (parents[0]) -> project root as parents[0].parents[0]
    # But to be safe, walk up until we find the project folder containing this package
    # Insert the project root (two levels up) if not present
    PROJ_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJ_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJ_ROOT))
    from Tairs_Scripts_norm.config import TEST_XLSX, OUTPUT_BASE

# Visualization config
SEED = 42
MAX_INSTANCES_PER_IMAGE = 20
IMG_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def label_to_image_path_from_labelfile(label_file: str) -> Path:
    # The label_file entries in the original CSV are full paths to label txt under the YOLO test set.
    # Try derive image path by replacing labels/ with images/ and suffix
    lf = Path(label_file)
    # Expected test images are under the same TEST_XLSX's dataset folder: we assume prawn_2025_circ_small_v1 structure
    test_root = Path(TEST_XLSX).resolve().parent
    labels_dir = test_root / "labels"
    images_dir = test_root / "images"

    try:
        rel = lf.relative_to(labels_dir)
    except Exception:
        rel = lf.name

    stem_rel = Path(rel).with_suffix("")

    for ext in IMG_EXTS:
        cand = images_dir / (str(stem_rel) + ext)
        if cand.exists():
            return cand

    for ext in IMG_EXTS:
        cand = images_dir / (stem_rel.name + ext)
        if cand.exists():
            return cand

    raise FileNotFoundError(f"Could not find matching image for label: {label_file}")


def draw_point(img, x_px: int, y_px: int, color, text: str) -> None:
    cv2.circle(img, (x_px, y_px), 6, color, -1)
    cv2.putText(img, text, (x_px + 8, y_px - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def visualize_model(csv_path: Path) -> None:
    if not csv_path.exists():
        return

    print(f"Processing model: {csv_path.name}")

    out_dir = csv_path.parent / f"viz_{csv_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Choose columns: prefer body-frame (e_l/e_d), but visualization here overlays on image which is pixel coords.
    # If body-frame, we cannot overlay directly. So prefer image coords if available; otherwise skip image overlay and
    # save body-frame scatter plots.
    if {"e_x", "e_y", "e_pred_x", "e_pred_y"}.issubset(set(df.columns)):
        # Image overlay path
        use_image_overlay = True
    else:
        use_image_overlay = False

    # If evaluation summary exists, use it to pick worst images
    summary_csv = csv_path.parent / f"evaluation_summary_{csv_path.stem}.csv"
    chosen_labels = []

    if summary_csv.exists():
        try:
            summ_df = pd.read_csv(summary_csv)
            worst = summ_df[summ_df["group"] == "worst_images_meanDist"]
            if not worst.empty:
                chosen_labels = worst["label_file"].astype(str).tolist()[:5]
                print(f"  Using {len(chosen_labels)} worst images from summary.")
        except Exception as e:
            print(f"  Could not read summary {summary_csv.name}: {e}")

    # If we have image coords, proceed with image overlay selection
    if use_image_overlay:
        if not chosen_labels:
            if "label_file" in df.columns:
                unique_labels = df["label_file"].dropna().unique().tolist()
                if len(unique_labels) > 0:
                    rng = np.random.default_rng(SEED)
                    count = min(len(unique_labels), 2)
                    chosen_labels = rng.choice(unique_labels, size=count, replace=False).tolist()
                    print(f"  Using {len(chosen_labels)} random images (no summary found).")

        if not chosen_labels:
            print("  No labels to visualize (no label_file entries found).")
            return

        for idx, lf_str in enumerate(chosen_labels, start=1):
            try:
                img_path = label_to_image_path_from_labelfile(lf_str)
            except FileNotFoundError as e:
                print(f"  [Warn] {e}")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [Warn] Could not read image: {img_path}")
                continue

            h, w = img.shape[:2]

            sub = df[df["label_file"] == lf_str].copy()
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
    # If image overlay was used, we've already written images and can return
    if use_image_overlay:
        print(f"  Saved visualizations to: {out_dir}")
        return

    # ------------------ BODY-FRAME VISUALIZATIONS ------------------
    # Create body-frame scatter plots when image overlay isn't possible
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("  Matplotlib not available; skipping body-frame visualizations.")
        return

    # overall scatter (GT vs Pred) in body-frame
    mask = (~df.get("e_pred_l", pd.Series(dtype=float)).isna()) & (~df.get("e_pred_d", pd.Series(dtype=float)).isna())
    has_gt = ("e_l" in df.columns) and ("e_d" in df.columns)

    fig, ax = plt.subplots(figsize=(6, 6))
    if has_gt:
        ax.scatter(df.loc[mask, "e_l"], df.loc[mask, "e_d"], label="GT", s=30, alpha=0.6)
    ax.scatter(df.loc[mask, "e_pred_l"], df.loc[mask, "e_pred_d"], label="Pred", s=20, alpha=0.8)
    ax.axhline(0, linestyle="--", color="gray")
    ax.set_xlabel("l_norm")
    ax.set_ylabel("d_norm")
    ax.set_title(f"Body-frame scatter (all) | {csv_path.stem}")
    ax.legend()
    out_path = out_dir / f"bodyframe_scatter_all_{csv_path.stem}.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    # Per-object worst scatter plots if image/object identifiers exist
    if "image_stem" in df.columns and "object_id" in df.columns and "dist" in df.columns:
        grp = df.groupby(["image_stem", "object_id"], as_index=False)["dist"].mean()
        worst_objs = grp.sort_values("dist", ascending=False).head(5)
        for idx, row in enumerate(worst_objs.itertuples(index=False), start=1):
            img_stem = row.image_stem
            obj_id = row.object_id
            sub = df[(df["image_stem"] == img_stem) & (df["object_id"] == obj_id)]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(6, 6))
            if has_gt:
                ax.scatter(sub["e_l"], sub["e_d"], label="GT", s=40)
            ax.scatter(sub["e_pred_l"], sub["e_pred_d"], label="Pred", s=30)
            ax.axhline(0, linestyle="--", color="gray")
            ax.set_xlabel("l_norm")
            ax.set_ylabel("d_norm")
            safe_name = str(img_stem).replace('/', '_')
            ax.set_title(f"Body-frame scatter | {csv_path.stem} | {safe_name} obj{obj_id}")
            ax.legend()
            out_path = out_dir / f"bodyframe_scatter_{idx}_{safe_name}_obj{obj_id}.png"
            fig.tight_layout()
            fig.savefig(out_path)
            plt.close(fig)

    print(f"  Saved visualizations to: {out_dir}")


def main() -> None:
    # Aggregate model CSVs from multiple possible model roots
    models_roots = []
    if OUTPUT_BASE is not None:
        models_roots.append(Path(OUTPUT_BASE))
    # original Tairs_Scripts models folder (project relative)
    proj_root = Path(__file__).resolve().parents[1]
    other_root = proj_root / "Tairs_Scripts" / "models_csv" / "models"
    models_roots.append(other_root)

    csv_set = []
    for root in models_roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.csv")):
            # skip evaluation summaries and feature files by default
            if p.name.startswith("evaluation_summary_") or p.name.startswith("features_"):
                continue
            if p not in csv_set:
                csv_set.append(p)

    if not csv_set:
        print(f"No model CSVs found in any of: {models_roots}")
        return

    for csv_path in csv_set:
        try:
            visualize_model(csv_path)
        except Exception as e:
            print(f"Failed to visualize {csv_path}: {e}")


if __name__ == "__main__":
    main()
