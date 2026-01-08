#!/usr/bin/env python3
"""
Run identical focused EDA on multiple keypoint Excel files.

Datasets:
1. train_on_all
2. yolo_val (prawn_2025_circ_small_v1)

EDA includes:
- Dataset-level overview
- Annotation validation
- Visibility / completeness
- Anatomically meaningful distances only
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# ============================================================
# DATASETS (FIXED PATHS)
# ============================================================
DATASETS = {
    "train_on_all": Path(
        "/Users/taircarmon/Desktop/ml-course-project/train_on_all/keypoints_from_label_txt.xlsx"
    ),
    "yolo_val": Path(
        "/Users/taircarmon/Desktop/ml-course-project/prawn_2025_circ_small_v1/keypoints_from_label_txt_test.xlsx"
    ),
}

BASE_OUTDIR = Path(
    "/Users/taircarmon/Desktop/ml-course-project/exploratory_analysis/outputs"
)
BASE_OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# ANATOMICAL DEFINITIONS
# ============================================================
ANATOMICAL_KP = {
    0: "carapace",
    1: "eyes",
    2: "rostrum",
    3: "tail",
}

ANATOMICAL_PAIRS = [
    ("rostrum", "tail"),
    ("carapace", "rostrum"),
    ("carapace", "tail"),
    ("eyes", "carapace"),
]

# ============================================================
# HELPERS
# ============================================================
def pivot_long_to_wide(
    df,
    keypoint_index_col="keypoint_index",
    xcol="x_norm",
    ycol="y_norm",
    visibility_col="visibility",
):
    exclude = {keypoint_index_col, xcol, ycol, visibility_col}
    group_cols = [c for c in df.columns if c not in exclude]

    df = df.copy()
    df[keypoint_index_col] = df[keypoint_index_col].astype(int)

    x_p = df.pivot_table(index=group_cols, columns=keypoint_index_col, values=xcol, aggfunc="first")
    y_p = df.pivot_table(index=group_cols, columns=keypoint_index_col, values=ycol, aggfunc="first")
    v_p = df.pivot_table(index=group_cols, columns=keypoint_index_col, values=visibility_col, aggfunc="first")

    wide = pd.DataFrame(index=x_p.index)
    for k in x_p.columns:
        wide[f"kp{k}_x"] = pd.to_numeric(x_p[k], errors="coerce")
        wide[f"kp{k}_y"] = pd.to_numeric(y_p[k], errors="coerce")
        wide[f"kp{k}_vis"] = v_p[k]

    return wide.reset_index()


# ============================================================
# EDA PIPELINE (RUN PER DATASET)
# ============================================================
def run_eda(df: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 0. BASIC DATASET OVERVIEW
    # --------------------------------------------------------
    overview = {
        "n_rows": len(df),
        "n_images": df["image_stem"].nunique(),
        "n_objects": df[["image_stem", "object_id"]].drop_duplicates().shape[0],
        "n_keypoints_total": len(df),
        "n_keypoint_types": df["keypoint_index"].nunique(),
    }

    kp_per_object = df.groupby(["image_stem", "object_id"]).size()
    overview["kp_per_object_min"] = int(kp_per_object.min())
    overview["kp_per_object_max"] = int(kp_per_object.max())
    overview["kp_per_object_mean"] = float(kp_per_object.mean())
    overview["visibility_rate_mean"] = float((df["visibility"] > 0).mean())

    pd.DataFrame.from_dict(overview, orient="index", columns=["value"]) \
        .to_csv(outdir / "dataset_overview.csv")

    kp_counts = (
        df["keypoint_index"]
        .value_counts()
        .sort_index()
        .rename_axis("keypoint_index")
        .reset_index(name="count")
    )
    kp_counts["keypoint_name"] = kp_counts["keypoint_index"].map(ANATOMICAL_KP)
    kp_counts.to_csv(outdir / "keypoint_counts.csv", index=False)

    # --------------------------------------------------------
    # Pivot to wide
    # --------------------------------------------------------
    wide = pivot_long_to_wide(df)

    kp_indices = sorted(
        int(c.split("_")[0][2:])
        for c in wide.columns
        if c.startswith("kp") and c.endswith("_x")
    )

    coord_cols = [c for c in wide.columns if c.endswith("_x") or c.endswith("_y")]

    # --------------------------------------------------------
    # 1. VALIDATION
    # --------------------------------------------------------
    validation = pd.DataFrame(index=wide.index)
    validation["n_keypoints_expected"] = len(kp_indices)
    validation["n_missing_coords"] = wide[coord_cols].isna().sum(axis=1)
    validation["all_numeric"] = wide[coord_cols].notna().all(axis=1)
    validation["coords_in_0_1"] = wide[coord_cols].apply(
        lambda s: s.between(0, 1) | s.isna()
    ).all(axis=1)

    validation.to_csv(outdir / "validation_report.csv", index=False)

    # --------------------------------------------------------
    # 2. VISIBILITY PER KEYPOINT
    # --------------------------------------------------------
    vis_rows = []
    for k in kp_indices:
        name = ANATOMICAL_KP.get(k, f"kp{k}")
        vis = wide[f"kp{k}_vis"].fillna(0) > 0
        vis_rows.append({
            "keypoint": name,
            "visibility_rate": vis.mean(),
            "n_visible": int(vis.sum()),
            "n_missing": int((~vis).sum()),
        })

    pd.DataFrame(vis_rows).to_csv(outdir / "keypoint_visibility.csv", index=False)

    # --------------------------------------------------------
    # 3. ANATOMICAL DISTANCES
    # --------------------------------------------------------
    dist_rows = []
    for _, row in wide.iterrows():
        coords = {}
        for k in kp_indices:
            x = row.get(f"kp{k}_x")
            y = row.get(f"kp{k}_y")
            if pd.notna(x) and pd.notna(y):
                coords[ANATOMICAL_KP.get(k, f"kp{k}")] = np.array([x, y])

        for a, b in ANATOMICAL_PAIRS:
            if a in coords and b in coords:
                dist_rows.append({
                    "pair": f"{a}_{b}",
                    "distance": np.linalg.norm(coords[a] - coords[b]),
                })

    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(outdir / "anatomical_distances.csv", index=False)

    dist_df.groupby("pair")["distance"] \
        .agg(["mean", "std", "min", "max", "count"]) \
        .reset_index() \
        .to_csv(outdir / "anatomical_distance_stats.csv", index=False)

    # --------------------------------------------------------
    # 4. ROSTRUM–TAIL SANITY PLOT
    # --------------------------------------------------------
    rt = dist_df[dist_df["pair"] == "rostrum_tail"]["distance"].dropna()
    if not rt.empty:
        plt.figure(figsize=(6, 4))
        sns.histplot(rt, bins=40)
        plt.xlabel("Normalized distance")
        plt.title("Rostrum–Tail distance (image-normalized)")
        plt.tight_layout()
        plt.savefig(outdir / "rostrum_tail_distribution.png")
        plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    for name, xlsx_path in DATASETS.items():
        print(f"\nRunning EDA for: {name}")
        print(f"Source: {xlsx_path}")

        df = pd.read_excel(xlsx_path)

        required = {
            "image_stem",
            "object_id",
            "keypoint_index",
            "x_norm",
            "y_norm",
            "visibility",
        }
        if not required.issubset(df.columns):
            raise ValueError(f"{name}: missing required columns")

        run_eda(df, BASE_OUTDIR / name)

    print("\nEDA completed for all datasets.")
    print(f"Outputs written to:\n{BASE_OUTDIR}\n")


if __name__ == "__main__":
    main()
