#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import re

# ============================================================
# CONFIG
# ============================================================
KEYPOINTS_XLSX = Path("/train_on_all/keypoints_from_label_txt.xlsx")
IMAGES_DIR = Path("/Users/taircarmon/Desktop/ml-course-project/train_on_all")
OUTPUT_PATH = KEYPOINTS_XLSX.with_name("keypoints_body_frame.xlsx")

# ============================================================
# KEYPOINT INDICES (ANATOMICAL)
# ============================================================
KP_CARAPACE = 0
KP_EYES     = 1
KP_ROSTRUM  = 2
KP_TAIL     = 3

ROSTRUM_KP = KP_ROSTRUM
TAIL_KP    = KP_TAIL


# ============================================================
# HELPERS
# ============================================================
def extract_base_stem(image_stem: str) -> str:
    base = image_stem
    base = re.split(r"\.rf\.", base)[0]
    base = re.split(r"-jpg|_jpg", base)[0]
    return base

def find_image_recursive(base_stem: str, root: Path) -> Path:
    """
    Search recursively under root for any image file whose *stem* contains base_stem.
    Works with nested train/val folders etc.
    """
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    hits = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            if base_stem in p.stem:
                hits.append(p)
    if not hits:
        # Helpful debug: show a few filenames so we can see naming style
        sample = []
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                sample.append(p.name)
                if len(sample) >= 10:
                    break
        raise FileNotFoundError(
            f"No image found for base stem: {base_stem}\n"
            f"Searched recursively under: {root}\n"
            f"Example files I did see: {sample}"
        )
    # if multiple matches, pick the shortest filename (often the most direct one)
    hits.sort(key=lambda x: len(x.name))
    return hits[0]

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_excel(KEYPOINTS_XLSX)

required_cols = {"image_stem", "object_id", "keypoint_index", "x_norm", "y_norm", "visibility"}
assert required_cols.issubset(df.columns), f"Missing required columns. Have: {df.columns.tolist()}"

# ============================================================
# BODY FRAME TRANSFORMATION
# ============================================================
rows = []
for (img, obj), g in df.groupby(["image_stem", "object_id"]):
    kp = g.set_index("keypoint_index")
    if ROSTRUM_KP not in kp.index or TAIL_KP not in kp.index:
        continue

    r = kp.loc[ROSTRUM_KP, ["x_norm", "y_norm"]].values.astype(float)
    t = kp.loc[TAIL_KP, ["x_norm", "y_norm"]].values.astype(float)

    v = t - r
    L = np.linalg.norm(v)
    if L == 0:
        continue

    v_hat = v / L
    v_perp = np.array([-v_hat[1], v_hat[0]])

    for _, row in g.iterrows():
        p = np.array([row.x_norm, row.y_norm]) - r
        l = np.dot(p, v_hat)
        d = np.dot(p, v_perp)
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
body_df.to_excel(OUTPUT_PATH, index=False)
print(f"Saved transformed data to: {OUTPUT_PATH}")

# ============================================================
# VISUALIZATION (ONE IMAGE + OBJECT)
# ============================================================
example = body_df.iloc[0]
image_stem = str(example.image_stem)
object_id = int(example.object_id)

base_stem = extract_base_stem(image_stem)
print(f"image_stem: {image_stem}")
print(f"base_stem:  {base_stem}")

img_path = find_image_recursive(base_stem, IMAGES_DIR)
print(f"Using image: {img_path}")

orig = df[(df.image_stem == image_stem) & (df.object_id == object_id)].copy()
trans = body_df[(body_df.image_stem == image_stem) & (body_df.object_id == object_id)].copy()

# Load image
img = cv2.imread(str(img_path))
if img is None:
    raise RuntimeError(f"cv2.imread failed for: {img_path}")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape

# Pixel coords
orig["x_px"] = orig["x_norm"].astype(float) * w
orig["y_px"] = orig["y_norm"].astype(float) * h

kp = orig.set_index("keypoint_index")
r_px = kp.loc[ROSTRUM_KP, ["x_px", "y_px"]].values.astype(float)
t_px = kp.loc[TAIL_KP, ["x_px", "y_px"]].values.astype(float)

# Rotation to align body axis horizontally
v = t_px - r_px
angle = np.degrees(np.arctan2(v[1], v[0]))
M = cv2.getRotationMatrix2D(tuple(r_px), -angle, 1.0)
rot_img = cv2.warpAffine(img, M, (w, h))

pts = np.column_stack([orig["x_px"].values, orig["y_px"].values, np.ones(len(orig))])
rot_pts = (M @ pts.T).T

# Plot: image+keypoints, rotated image+keypoints, body-frame (l,d)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ax = axes[0]
ax.imshow(img)
ax.scatter(orig["x_px"], orig["y_px"], s=60, c="yellow", edgecolors="black")
ax.plot([r_px[0], t_px[0]], [r_px[1], t_px[1]], linewidth=2, label="Body axis (rostrum→tail)")
ax.set_title("Original image + keypoints")
ax.axis("off")
ax.legend()

ax = axes[1]
ax.imshow(rot_img)
ax.scatter(rot_pts[:, 0], rot_pts[:, 1], s=60, c="cyan", edgecolors="black")
ax.set_title("Body-aligned (rotated) image")
ax.axis("off")

ax = axes[2]
ax.scatter(trans["l_norm"], trans["d_norm"], s=70)
ax.axhline(0, linestyle="--")
ax.set_xlabel("Longitudinal (ℓ/L)")
ax.set_ylabel("Lateral (d/L)")
ax.set_title("Body-frame representation")
ax.axis("equal")

plt.tight_layout()
plt.show()