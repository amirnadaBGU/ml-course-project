#!/usr/bin/env python3
"""
Body-frame geometric data understanding for prawn keypoints.

Focus:
- Anatomical interpretation of keypoints
- Distribution along and across the body axis
- Keypoint-specific geometric variability
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# ============================================================
# PATHS
# ============================================================
INPUT_XLSX = Path(
    "/Users/taircarmon/Desktop/ml-course-project/train_on_all/keypoints_body_frame.xlsx"
)

BASE_DIR = Path(
    "/Users/taircarmon/Desktop/ml-course-project/exploratory_analysis/after_body_norm"
)

OUTDIR = BASE_DIR / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# ANATOMICAL DEFINITIONS
# ============================================================
KP_NAMES = {
    0: "carapace",
    1: "eyes",
    2: "rostrum",
    3: "tail",
}

KP_ORDER = ["rostrum", "eyes", "carapace", "tail"]

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_excel(INPUT_XLSX)

required_cols = {
    "image_stem",
    "object_id",
    "keypoint_index",
    "l_norm",
    "d_norm",
}
if not required_cols.issubset(df.columns):
    raise ValueError(
        f"Missing required columns. Found: {df.columns.tolist()}"
    )

df["keypoint"] = df["keypoint_index"].map(KP_NAMES)

# ============================================================
# 1. LONGITUDINAL DISTRIBUTION (ℓ / L)
# ============================================================
plt.figure(figsize=(8, 5))
sns.violinplot(
    data=df,
    x="keypoint",
    y="l_norm",
    order=KP_ORDER,
    inner="quartile",
)
plt.axhline(0, linestyle="--", linewidth=1)
plt.axhline(1, linestyle="--", linewidth=1)
plt.ylabel("Longitudinal coordinate (ℓ / L)")
plt.xlabel("")
plt.title("Longitudinal body-frame distribution by keypoint")
plt.tight_layout()
plt.savefig(OUTDIR / "longitudinal_distribution_by_keypoint.png")
plt.close()

# ============================================================
# 2. LATERAL DISTRIBUTION (d / L)
# ============================================================
plt.figure(figsize=(8, 5))
sns.violinplot(
    data=df,
    x="keypoint",
    y="d_norm",
    order=KP_ORDER,
    inner="quartile",
)
plt.axhline(0, linestyle="--", linewidth=1)
plt.ylabel("Lateral coordinate (d / L)")
plt.xlabel("")
plt.title("Lateral body-frame distribution by keypoint")
plt.tight_layout()
plt.savefig(OUTDIR / "lateral_distribution_by_keypoint.png")
plt.close()

# ============================================================
# 3. GEOMETRIC VARIABILITY SUMMARY
# ============================================================
summary = (
    df.groupby("keypoint")[["l_norm", "d_norm"]]
    .agg(["mean", "std"])
    .reset_index()
)

summary.columns = [
    "keypoint",
    "l_mean",
    "l_std",
    "d_mean",
    "d_std",
]

summary.to_csv(
    OUTDIR / "keypoint_geometric_summary.csv",
    index=False
)

# ============================================================
# 4. BODY-FRAME SCATTER (ℓ/L vs d/L)
# ============================================================
plt.figure(figsize=(6, 5))
sns.scatterplot(
    data=df,
    x="l_norm",
    y="d_norm",
    hue="keypoint",
    alpha=0.3,
    s=12,
)
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Longitudinal (ℓ / L)")
plt.ylabel("Lateral (d / L)")
plt.title("Body-frame keypoint geometry")
plt.axis("equal")
plt.legend(
    title="Keypoint",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)
plt.tight_layout()
plt.savefig(OUTDIR / "body_frame_scatter.png")
plt.close()

print("Body-frame geometric EDA completed successfully.")
print(f"All outputs written to:\n{OUTDIR}")
