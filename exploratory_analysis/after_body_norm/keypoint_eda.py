#!/usr/bin/env python3
"""
Per-keypoint geometric EDA in body-frame coordinates.

For each keypoint:
- Histogram of longitudinal coordinate (ℓ / L)
- Histogram of lateral coordinate (d / L)
- Scatter plot: ℓ / L vs d / L
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

OUTDIR = BASE_DIR / "outputs" / "per_keypoint"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# KEYPOINT DEFINITIONS
# ============================================================
KP_NAMES = {
    0: "carapace",
    1: "eyes",
    2: "rostrum",
    3: "tail",
}

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_excel(INPUT_XLSX)

required_cols = {
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
# PER-KEYPOINT ANALYSIS
# ============================================================
for kp_name in KP_NAMES.values():
    kp_df = df[df["keypoint"] == kp_name]

    kp_outdir = OUTDIR / kp_name
    kp_outdir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Histogram: Longitudinal coordinate (ℓ / L)
    # --------------------------------------------------------
    plt.figure(figsize=(6, 4))
    sns.histplot(
        kp_df["l_norm"],
        bins=50,
        kde=True
    )
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.axvline(1, linestyle="--", linewidth=1)
    plt.xlabel("Longitudinal coordinate (ℓ / L)")
    plt.ylabel("Count")
    plt.title(f"{kp_name}: longitudinal distribution")
    plt.tight_layout()
    plt.savefig(kp_outdir / "hist_l.png")
    plt.close()

    # --------------------------------------------------------
    # 2. Histogram: Lateral coordinate (d / L)
    # --------------------------------------------------------
    plt.figure(figsize=(6, 4))
    sns.histplot(
        kp_df["d_norm"],
        bins=50,
        kde=True
    )
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Lateral coordinate (d / L)")
    plt.ylabel("Count")
    plt.title(f"{kp_name}: lateral distribution")
    plt.tight_layout()
    plt.savefig(kp_outdir / "hist_d.png")
    plt.close()

    # --------------------------------------------------------
    # 3. Scatter: ℓ / L vs d / L
    # --------------------------------------------------------
    plt.figure(figsize=(5, 5))
    sns.scatterplot(
        x=kp_df["l_norm"],
        y=kp_df["d_norm"],
        alpha=0.3,
        s=10
    )
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.axvline(1, linestyle="--", linewidth=1)
    plt.xlabel("Longitudinal (ℓ / L)")
    plt.ylabel("Lateral (d / L)")
    plt.title(f"{kp_name}: body-frame geometry")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(kp_outdir / "scatter_ld.png")
    plt.close()

    print(f"Finished EDA for keypoint: {kp_name}")

print("Per-keypoint geometric EDA completed successfully.")
print(f"All outputs written to:\n{OUTDIR}")
