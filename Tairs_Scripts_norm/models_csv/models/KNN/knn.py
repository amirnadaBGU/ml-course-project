from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
try:
    from Tairs_Scripts_norm.config import (
        TRAIN_XLSX, TEST_XLSX, OUTPUT_BASE,
        KP_CARAPACE, KP_EYES, KP_ROSTRUM, KP_TAIL,
        NUM_KPTS, MIN_VIS
    )
except Exception:
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[4]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from Tairs_Scripts_norm.config import (
        TRAIN_XLSX, TEST_XLSX, OUTPUT_BASE,
        KP_CARAPACE, KP_EYES, KP_ROSTRUM, KP_TAIL,
        NUM_KPTS, MIN_VIS
    )

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "KNN"
OUT_DIR = OUTPUT_BASE / MODEL_NAME
OUT_CSV  = OUT_DIR / "knn_eyes_predictions.csv"
OUT_XLSX = OUT_DIR / "knn_eyes_predictions.xlsx"

# KNN Hyperparameters
N_NEIGHBORS = 5


# ============================================================
# Helpers
# ============================================================

def load_bodyframe_excel(xlsx: Path) -> pd.DataFrame:
    if not xlsx.exists():
        raise FileNotFoundError(f"Excel not found: {xlsx}")
    df = pd.read_excel(xlsx)
    return df


# ============================================================
# MAIN
# ============================================================

def main():
    print("=== KNN Model on normalized keypoints: Eyes Prediction ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_train = load_bodyframe_excel(TRAIN_XLSX)
    df_test = load_bodyframe_excel(TEST_XLSX)

    # ------------------------------------------------------------
    # 1) Build TRAIN set
    # ------------------------------------------------------------
    X_train = []
    Y_train = []

    grouped = df_train.groupby(["image_stem", "object_id"])
    for (img, obj), g in grouped:
        kp = g.set_index("keypoint_index")
        # Require all 4 keypoints visible for training
        if not all(k in kp.index for k in [KP_CARAPACE, KP_EYES, KP_ROSTRUM, KP_TAIL]):
            continue
        if min(kp.loc[KP_CARAPACE, "visibility"], kp.loc[KP_EYES, "visibility"], kp.loc[KP_ROSTRUM, "visibility"], kp.loc[KP_TAIL, "visibility"]) < MIN_VIS:
            continue

        X_train.append([
            kp.loc[KP_CARAPACE, "l_norm"], kp.loc[KP_CARAPACE, "d_norm"],
            kp.loc[KP_ROSTRUM, "l_norm"],  kp.loc[KP_ROSTRUM, "d_norm"],
            kp.loc[KP_TAIL, "l_norm"],     kp.loc[KP_TAIL, "d_norm"],
        ])
        Y_train.append([
            kp.loc[KP_EYES, "l_norm"], kp.loc[KP_EYES, "d_norm"],
        ])

    if len(X_train) < 10:
        raise RuntimeError(f"Not enough training instances (found {len(X_train)}).")

    X_train = np.array(X_train, dtype=float)
    Y_train = np.array(Y_train, dtype=float)

    print(f"Training samples: {len(X_train)}")

    # ------------------------------------------------------------
    # 2) Train KNN Model
    # ------------------------------------------------------------
    print(f"Training KNN (k={N_NEIGHBORS})...")
    model = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=N_NEIGHBORS)
    )
    model.fit(X_train, Y_train)

    # ------------------------------------------------------------
    # 3) Predict on TEST set
    # ------------------------------------------------------------
    rows = []
    grouped_test = df_test.groupby(["image_stem", "object_id"])
    for (img, obj), g in grouped_test:
        kp = g.set_index("keypoint_index")
        if not all(k in kp.index for k in [KP_CARAPACE, KP_ROSTRUM, KP_TAIL]):
            continue
        if min(kp.loc[KP_CARAPACE, "visibility"], kp.loc[KP_ROSTRUM, "visibility"], kp.loc[KP_TAIL, "visibility"]) < MIN_VIS:
            continue

        X_in = np.array([[
            kp.loc[KP_CARAPACE, "l_norm"], kp.loc[KP_CARAPACE, "d_norm"],
            kp.loc[KP_ROSTRUM, "l_norm"],  kp.loc[KP_ROSTRUM, "d_norm"],
            kp.loc[KP_TAIL, "l_norm"],     kp.loc[KP_TAIL, "d_norm"],
        ]], dtype=float)

        pred = model.predict(X_in)[0]
        e_pred_l, e_pred_d = float(pred[0]), float(pred[1])

        has_gt_eyes = (KP_EYES in kp.index and kp.loc[KP_EYES, "visibility"] >= MIN_VIS)
        if has_gt_eyes:
            ex = float(kp.loc[KP_EYES, "l_norm"])
            ey = float(kp.loc[KP_EYES, "d_norm"])
            dist = float(np.sqrt((ex - e_pred_l) ** 2 + (ey - e_pred_d) ** 2))
            abs_err_l = abs(ex - e_pred_l)
            abs_err_d = abs(ey - e_pred_d)
        else:
            ex = np.nan
            ey = np.nan
            dist = np.nan
            abs_err_l = np.nan
            abs_err_d = np.nan

        rows.append({
            "image_stem": img,
            "object_id": obj,
            "e_l": ex,
            "e_d": ey,
            "e_v": float(kp.loc[KP_EYES, "visibility"]) if KP_EYES in kp.index else 0,
            "e_pred_l": e_pred_l,
            "e_pred_d": e_pred_d,
            "abs_err_l": abs_err_l,
            "abs_err_d": abs_err_d,
            "dist": dist,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    df_out.to_excel(OUT_XLSX, index=False)

    # Quick console summary
    eval_df = df_out[df_out["e_v"] >= MIN_VIS]
    if not eval_df.empty:
        mae_l = eval_df["abs_err_l"].mean()
        mae_d = eval_df["abs_err_d"].mean()
        dist_avg = eval_df["dist"].mean()
        print("\n--- Test Metrics (GT eyes visible) ---")
        print(f"MAE_l    : {mae_l:.6f}")
        print(f"MAE_d    : {mae_d:.6f}")
        print(f"Dist_avg : {dist_avg:.6f}")

    print(f"\nSaved results to:\n{OUT_CSV}\n{OUT_XLSX}")


if __name__ == "__main__":
    main()
