from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

try:
    # Preferred when used as a package: python -m Tairs_Scripts_norm.models_csv.models.BaseLine.Baseline
    from Tairs_Scripts_norm.config import (
        TRAIN_XLSX, TEST_XLSX, OUTPUT_BASE,
        KP_CARAPACE, KP_EYES, KP_ROSTRUM, KP_TAIL,
        NUM_KPTS, MIN_VIS
    )
except Exception:
    # Fallback when executed directly: python path/to/Baseline.py
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
MODEL_NAME = "BaseLine"
OUT_DIR = OUTPUT_BASE / MODEL_NAME
OUT_BASENAME = "baseline_linreg_eyes_from3kpts_test"
OUT_CSV  = OUT_DIR / f"{OUT_BASENAME}.csv"
OUT_XLSX = OUT_DIR / f"{OUT_BASENAME}.xlsx"


# ============================================================
# HELPERS
# ============================================================

def load_bodyframe_excel(xlsx: Path) -> pd.DataFrame:
    if not xlsx.exists():
        raise FileNotFoundError(f"Excel not found: {xlsx}")
    df = pd.read_excel(xlsx)
    return df


def fit_linear_regression(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X1 = np.hstack([np.ones((X.shape[0], 1)), X])
    B, *_ = np.linalg.lstsq(X1, Y, rcond=None)
    return B


def predict_linear_regression(B: np.ndarray, X: np.ndarray) -> np.ndarray:
    X1 = np.hstack([np.ones((X.shape[0], 1)), X])
    return X1 @ B


# ============================================================
# MAIN
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_train = load_bodyframe_excel(TRAIN_XLSX)
    df_test = load_bodyframe_excel(TEST_XLSX)

    # Build training arrays: group by image/object and extract keypoints
    X_train = []
    Y_train = []

    grouped = df_train.groupby(["image_stem", "object_id"])
    for (img, obj), g in grouped:
        kp = g.set_index("keypoint_index")
        # require all 4 keypoints visible
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

    X_train = np.array(X_train, dtype=float)
    Y_train = np.array(Y_train, dtype=float)

    if len(X_train) < 10:
        raise RuntimeError("Not enough training samples in normalized keypoints.")

    print(f"[INFO] Training samples used: {len(X_train)}")

    B = fit_linear_regression(X_train, Y_train)

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------
    rows = []
    grouped_test = df_test.groupby(["image_stem", "object_id"])
    for (img, obj), g in grouped_test:
        kp = g.set_index("keypoint_index")
        if not all(k in kp.index for k in [KP_CARAPACE, KP_ROSTRUM, KP_TAIL]):
            continue
        if min(kp.loc[KP_CARAPACE, "visibility"], kp.loc[KP_ROSTRUM, "visibility"], kp.loc[KP_TAIL, "visibility"]) < MIN_VIS:
            continue

        X = np.array([[
            kp.loc[KP_CARAPACE, "l_norm"], kp.loc[KP_CARAPACE, "d_norm"],
            kp.loc[KP_ROSTRUM, "l_norm"],  kp.loc[KP_ROSTRUM, "d_norm"],
            kp.loc[KP_TAIL, "l_norm"],     kp.loc[KP_TAIL, "d_norm"],
        ]])

        pred = predict_linear_regression(B, X)[0]

        has_gt = (KP_EYES in kp.index and kp.loc[KP_EYES, "visibility"] >= MIN_VIS)
        if has_gt:
            gt = kp.loc[KP_EYES, ["l_norm", "d_norm"]].values.astype(float)
            dist = float(np.linalg.norm(pred - gt))
        else:
            dist = np.nan

        rows.append({
            "image_stem": img,
            "object_id": obj,
            "e_l": float(kp.loc[KP_EYES, "l_norm"]) if KP_EYES in kp.index else np.nan,
            "e_d": float(kp.loc[KP_EYES, "d_norm"]) if KP_EYES in kp.index else np.nan,
            "e_pred_l": float(pred[0]),
            "e_pred_d": float(pred[1]),
            "dist": dist,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    df_out.to_excel(OUT_XLSX, index=False)

    print(f"[OK] Saved results to:\n{OUT_CSV}\n{OUT_XLSX}")


if __name__ == "__main__":
    main()
