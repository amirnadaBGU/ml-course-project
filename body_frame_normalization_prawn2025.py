#!/usr/bin/env python3
"""
Create body-frame normalized keypoint Excel for the prawn_2025_circ_small_v1 YOLO dataset.

Default behavior:
 - Reads keypoints from: /Users/taircarmon/Desktop/ml-course-project/keypoints_from_label_txt.xlsx
 - Searches images under: /Users/taircarmon/Desktop/ml-course-project/prawn_2025_circ_small_v1/images
 - Writes output Excel to: /Users/taircarmon/Desktop/ml-course-project/prawn_2025_circ_small_v1/keypoints_body_frame.xlsx

Usage:
 python body_frame_normalization_prawn2025.py  # uses defaults
 python body_frame_normalization_prawn2025.py --keypoints /path/to/keypoints.xlsx --images /path/to/images --output /path/to/out.xlsx

This is based on your existing `body_frame_normalization.py` but targeted at the YOLO dataset folder.
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd

# Optional visualization (disabled by default to keep script fast/headless-friendly)
try:
    import cv2
    import matplotlib.pyplot as plt
    HAS_VIS = True
except Exception:
    HAS_VIS = False


def extract_base_stem(image_stem: str) -> str:
    base = image_stem
    base = re.split(r"\.rf\.", base)[0]
    base = re.split(r"-jpg|_jpg", base)[0]
    return base


def find_image_recursive(base_stem: str, root: Path) -> Path:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    hits = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            if base_stem in p.stem:
                hits.append(p)
    if not hits:
        raise FileNotFoundError(f"No image found for base stem: {base_stem} under {root}")
    hits.sort(key=lambda x: len(x.name))
    return hits[0]


def compute_body_frame(df, rostrum_kp=2, tail_kp=3):
    required_cols = {"image_stem", "object_id", "keypoint_index", "x_norm", "y_norm", "visibility"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns. Have: {df.columns.tolist()}")

    rows = []
    for (img, obj), g in df.groupby(["image_stem", "object_id"]):
        kp = g.set_index("keypoint_index")
        if rostrum_kp not in kp.index or tail_kp not in kp.index:
            # skip objects without rostrum or tail
            continue

        r = kp.loc[rostrum_kp, ["x_norm", "y_norm"]].values.astype(float)
        t = kp.loc[tail_kp, ["x_norm", "y_norm"]].values.astype(float)

        v = t - r
        L = np.linalg.norm(v)
        if L == 0 or np.isnan(L):
            continue

        v_hat = v / L
        v_perp = np.array([-v_hat[1], v_hat[0]])

        for _, row in g.iterrows():
            p = np.array([float(row.x_norm), float(row.y_norm)]) - r
            l = float(np.dot(p, v_hat))
            d = float(np.dot(p, v_perp))
            rows.append({
                "image_stem": img,
                "object_id": obj,
                "keypoint_index": int(row.keypoint_index),
                "visibility": row.visibility,
                "body_length": float(L),
                "l_norm": float(l / L),
                "d_norm": float(d / L),
            })

    body_df = pd.DataFrame(rows)
    return body_df


def main():
    p = argparse.ArgumentParser(description="Body-frame normalization for prawn_2025_circ_small_v1 dataset")
    p.add_argument("--keypoints", type=Path,
                   default=Path("/train_on_all/keypoints_from_label_txt.xlsx"),
                   help="Excel file with keypoints (default: root keypoints_from_label_txt.xlsx)")
    p.add_argument("--images", type=Path,
                   default=Path("/Users/taircarmon/Desktop/ml-course-project/prawn_2025_circ_small_v1/images"),
                   help="Root images directory for the YOLO dataset (default: prawn_2025_circ_small_v1/images)")
    p.add_argument("--output", type=Path,
                   default=Path("/Users/taircarmon/Desktop/ml-course-project/prawn_2025_circ_small_v1/keypoints_body_frame.xlsx"),
                   help="Output Excel path (default: saved inside the YOLO folder)")
    p.add_argument("--rostrum-kp", type=int, default=2, help="Keypoint index for rostrum (default 2)")
    p.add_argument("--tail-kp", type=int, default=3, help="Keypoint index for tail (default 3)")
    p.add_argument("--visualize", action="store_true", help="Show a small visualization of one example (requires opencv+matplotlib)")

    args = p.parse_args()

    if not args.keypoints.exists():
        raise FileNotFoundError(f"Keypoints file not found: {args.keypoints}")
    if not args.images.exists():
        raise FileNotFoundError(f"Images directory not found: {args.images}")

    print(f"Loading keypoints from: {args.keypoints}")
    df = pd.read_excel(args.keypoints)

    print("Computing body-frame normalized coordinates...")
    body_df = compute_body_frame(df, rostrum_kp=args.rostrum_kp, tail_kp=args.tail_kp)

    out_parent = args.output.parent
    out_parent.mkdir(parents=True, exist_ok=True)
    body_df.to_excel(args.output, index=False)
    print(f"Saved transformed data to: {args.output}")

    if args.visualize:
        if not HAS_VIS:
            print("Visualization requested but OpenCV/matplotlib not available. Skipping visualization.")
            return
        if body_df.empty:
            print("No entries to visualize.")
            return

        example = body_df.iloc[0]
        image_stem = str(example.image_stem)
        object_id = int(example.object_id)

        base_stem = extract_base_stem(image_stem)
        print(f"example image_stem: {image_stem} -> base: {base_stem}")
        try:
            img_path = find_image_recursive(base_stem, args.images)
        except FileNotFoundError as e:
            print(str(e))
            return

        # load original keypoints for that object
        orig = df[(df.image_stem == image_stem) & (df.object_id == object_id)].copy()
        if orig.empty:
            print("Original keypoints for the example were not found in the keypoints file.")
            return

        import cv2
        import matplotlib.pyplot as plt

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"cv2 failed to read image: {img_path}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        orig["x_px"] = orig["x_norm"].astype(float) * w
        orig["y_px"] = orig["y_norm"].astype(float) * h

        kp = orig.set_index("keypoint_index")
        if args.rostrum_kp not in kp.index or args.tail_kp not in kp.index:
            print("Example object does not contain rostrum/tail keypoints. Skipping vis.")
            return

        r_px = kp.loc[args.rostrum_kp, ["x_px", "y_px"]].values.astype(float)
        t_px = kp.loc[args.tail_kp, ["x_px", "y_px"]].values.astype(float)

        v = t_px - r_px
        angle = np.degrees(np.arctan2(v[1], v[0]))
        M = cv2.getRotationMatrix2D(tuple(r_px), -angle, 1.0)
        rot_img = cv2.warpAffine(img, M, (w, h))

        pts = np.column_stack([orig["x_px"].values, orig["y_px"].values, np.ones(len(orig))])
        rot_pts = (M @ pts.T).T

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        ax = axes[0]
        ax.imshow(img)
        ax.scatter(orig["x_px"], orig["y_px"], s=60, c="yellow", edgecolors="black")
        ax.plot([r_px[0], t_px[0]], [r_px[1], t_px[1]], linewidth=2)
        ax.set_title("Original image + keypoints")
        ax.axis("off")

        ax = axes[1]
        ax.imshow(rot_img)
        ax.scatter(rot_pts[:, 0], rot_pts[:, 1], s=60, c="cyan", edgecolors="black")
        ax.set_title("Body-aligned image")
        ax.axis("off")

        trans = body_df[(body_df.image_stem == image_stem) & (body_df.object_id == object_id)].copy()
        ax = axes[2]
        ax.scatter(trans["l_norm"], trans["d_norm"], s=70)
        ax.axhline(0, linestyle="--")
        ax.set_xlabel("Longitudinal (ℓ/L)")
        ax.set_ylabel("Lateral (d/L)")
        ax.set_title("Body-frame representation")
        ax.axis("equal")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

