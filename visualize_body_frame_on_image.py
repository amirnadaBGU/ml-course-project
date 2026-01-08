#!/usr/bin/env python3

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
IMAGE_PATH = Path(
    "/Users/taircarmon/Desktop/ml-course-project/prawn_2025_circ_small_v1/images/train/"
    "frame_00065_jpg.rf.abf3b996f5871335b15039d8481ea155.jpg"
)

LABEL_PATH = Path(
    "/Users/taircarmon/Desktop/ml-course-project/prawn_2025_circ_small_v1/labels/train/"
    "frame_00065_jpg.rf.abf3b996f5871335b15039d8481ea155.txt"
)

BODY_XLSX = Path(
    "/Users/taircarmon/Desktop/ml-course-project/prawn_2025_circ_small_v1/keypoints_body_frame.xlsx"
)

# ============================================================
# KEYPOINT DEFINITIONS
# ============================================================
KP_CARAPACE = 0
KP_EYES     = 1
KP_ROSTRUM  = 2
KP_TAIL     = 3

FOCUS_KP = KP_EYES  # <<< רק העין

# ============================================================
# LOAD IMAGE
# ============================================================
img = cv2.imread(str(IMAGE_PATH))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape

# ============================================================
# LOAD YOLO KEYPOINT LABEL
# ============================================================
with open(LABEL_PATH) as f:
    vals = list(map(float, f.readline().split()))

kp_vals = vals[5:]  # skip bbox

kps = []
for i in range(0, len(kp_vals), 3):
    kps.append([
        kp_vals[i] * w,
        kp_vals[i + 1] * h,
        int(kp_vals[i + 2])
    ])
kps = np.array(kps)

# ============================================================
# BODY AXES
# ============================================================
r = kps[KP_ROSTRUM, :2]
t = kps[KP_TAIL, :2]

v = t - r
L = np.linalg.norm(v)

u_x = v / L
u_y = np.array([-u_x[1], u_x[0]])

# ============================================================
# BODY FRAME COORDS FOR EYES (EXPLICIT)
# ============================================================
p_eye = kps[FOCUS_KP, :2] - r
l_eye = np.dot(p_eye, u_x)
d_eye = np.dot(p_eye, u_y)

l_eye_n = l_eye / L
d_eye_n = d_eye / L

# ============================================================
# ROTATION
# ============================================================
angle = np.degrees(np.arctan2(v[1], v[0]))
M = cv2.getRotationMatrix2D(tuple(r), -angle, 1.0)

rot_img = cv2.warpAffine(img, M, (w, h))
pts = np.column_stack([kps[:, :2], np.ones(len(kps))])
rot_pts = (M @ pts.T).T

# ============================================================
# BODY FRAME DATA (FOR CONTEXT)
# ============================================================
df_body = pd.read_excel(BODY_XLSX)
df_body = df_body[df_body.image_stem.str.contains("frame_00065")]

# ============================================================
# PLOTTING
# ============================================================
fig, axes = plt.subplots(1, 6, figsize=(32, 5))

# ------------------------------------------------------------
# 1. Image coordinate system (x,y) + eye
# ------------------------------------------------------------
ax = axes[0]
ax.imshow(img)

# image axes
ax.arrow(30, h - 30, 120, 0, head_width=10, color="black")
ax.arrow(30, h - 30, 0, -120, head_width=10, color="black")
ax.text(160, h - 35, "x", fontsize=12)
ax.text(25, h - 160, "y", fontsize=12)

# eye (small!)
ax.scatter(
    kps[FOCUS_KP, 0],
    kps[FOCUS_KP, 1],
    c="red",
    s=30,
)

ax.set_title("Image coordinate system (x,y)")
ax.axis("off")

# ------------------------------------------------------------
# 2. Body axes on image + eye
# ------------------------------------------------------------
ax = axes[1]
ax.imshow(img)

scale = 0.25 * L
ax.arrow(*r, *(u_x * scale), color="red", width=2)
ax.arrow(*r, *(u_y * scale), color="blue", width=2)

ax.scatter(
    kps[FOCUS_KP, 0],
    kps[FOCUS_KP, 1],
    c="red",
    s=30,
)

ax.text(*(r + u_x * scale * 1.05), r"$u_x$", color="red")
ax.text(*(r + u_y * scale * 1.05), r"$u_y$", color="blue")

ax.set_title("Body axes on image")
ax.axis("off")

# ------------------------------------------------------------
# 3. Body-aligned image + eye
# ------------------------------------------------------------
ax = axes[2]
ax.imshow(rot_img)
ax.scatter(
    rot_pts[FOCUS_KP, 0],
    rot_pts[FOCUS_KP, 1],
    c="red",
    s=30,
)
ax.set_title("Body-aligned image")
ax.axis("off")

# ------------------------------------------------------------
# 4. Body-frame coordinate system
# ------------------------------------------------------------
ax = axes[3]
ax.axhline(0, linestyle="--", color="gray")
ax.axvline(0, linestyle="--", color="gray")

ax.arrow(0, 0, 0.8, 0, head_width=0.02, color="red")
ax.arrow(0, 0, 0, 0.18, head_width=0.02, color="blue")

ax.text(0.82, 0.0, r"$u_x$", color="red")
ax.text(0.0, 0.20, r"$u_y$", color="blue")

ax.set_xlim(-0.2, 1.05)
ax.set_ylim(-0.3, 0.3)
ax.set_aspect("equal")

ax.set_title("Body-frame axes")
ax.set_xlabel("ℓ / L")
ax.set_ylabel("d / L")

# ------------------------------------------------------------
# 5. All keypoints in body frame (context)
# ------------------------------------------------------------
ax = axes[4]
ax.scatter(df_body.l_norm, df_body.d_norm, c="lightgray", s=25)
ax.axhline(0, linestyle="--", color="gray")
ax.axvline(0, linestyle="--", color="gray")

ax.set_xlim(-0.2, 1.05)
ax.set_ylim(-0.3, 0.3)
ax.set_aspect("equal")

ax.set_title("All keypoints (context)")
ax.set_xlabel("ℓ / L")
ax.set_ylabel("d / L")

# ------------------------------------------------------------
# 6. Eye only in body frame (small!)
# ------------------------------------------------------------
ax = axes[5]
ax.scatter(
    l_eye_n,
    d_eye_n,
    c="red",
    s=40,
)
ax.axhline(0, linestyle="--", color="gray")
ax.axvline(0, linestyle="--", color="gray")

ax.set_xlim(-0.2, 1.05)
ax.set_ylim(-0.3, 0.3)
ax.set_aspect("equal")

ax.set_title("Eyes keypoint (body frame)")
ax.set_xlabel("ℓ / L")
ax.set_ylabel("d / L")

plt.tight_layout()
plt.show()
