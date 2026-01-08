from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

# Find project root by folder name (SAFE & PORTABLE)
PROJECT_ROOT = next(
    p for p in SCRIPT_DIR.parents
    if p.name == "ml-course-project"
)

TRAIN_ROOT = PROJECT_ROOT / "train_on_all"
TEST_ROOT  = PROJECT_ROOT / "prawn_2025_circ_small_v1"

OUT_DIR = SCRIPT_DIR / "models_csv"
OUT_BASENAME = "baseline_linreg_eyes_from3kpts_test"
OUT_CSV  = OUT_DIR / f"{OUT_BASENAME}.csv"
OUT_XLSX = OUT_DIR / f"{OUT_BASENAME}.xlsx"

# ============================================================
# KEYPOINT CONFIG
# ============================================================
KP_CARAPACE = 0
KP_EYES     = 1
KP_ROSTRUM  = 2
KP_TAIL     = 3

NUM_KPTS = 4
MIN_VIS = 1   # visibility >= 1 is considered valid

# ============================================================
# HELPERS
# ============================================================
def list_label_files(root: Path) -> list[Path]:
    labels_dir = root / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing labels/ under: {root}")
    return sorted(
        p for p in labels_dir.rglob("*.txt")
        if p.is_file() and p.name != "labels.cache"
    )


def parse_yolo_pose_line(tokens: list[str]) -> dict:
    """
    YOLO pose format:
    cls x y w h  x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4
    """
    cls = int(tokens[0])
    box = list(map(float, tokens[1:5]))

    kpts_raw = list(map(float, tokens[5:5 + 3 * NUM_KPTS]))
    kpts = []
    for i in range(NUM_KPTS):
        x, y, v = kpts_raw[3*i:3*i+3]
        kpts.append((x, y, int(v)))

    return {
        "cls": cls,
        "box": box,
        "kpts": kpts
    }


def fit_linear_regression(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    X: (N, 6), Y: (N, 2)
    returns B: (7, 2) including intercept
    """
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

    # --------------------------------------------------------
    # TRAIN: eyes from (carapace, rostrum, tail)
    # --------------------------------------------------------
    X_train, Y_train = [], []

    for lf in list_label_files(TRAIN_ROOT):
        for line in lf.read_text().splitlines():
            if not line.strip():
                continue

            rec = parse_yolo_pose_line(line.split())
            k = rec["kpts"]

            # Require all 4 keypoints visible for training
            if min(k[KP_CARAPACE][2],
                   k[KP_EYES][2],
                   k[KP_ROSTRUM][2],
                   k[KP_TAIL][2]) < MIN_VIS:
                continue

            X_train.append([
                k[KP_CARAPACE][0], k[KP_CARAPACE][1],
                k[KP_ROSTRUM][0],  k[KP_ROSTRUM][1],
                k[KP_TAIL][0],     k[KP_TAIL][1],
            ])
            Y_train.append([
                k[KP_EYES][0],
                k[KP_EYES][1],
            ])

    X_train = np.array(X_train, dtype=float)
    Y_train = np.array(Y_train, dtype=float)

    if len(X_train) < 50:
        raise RuntimeError("Not enough training samples.")

    print(f"[INFO] Training samples used: {len(X_train)}")

    B = fit_linear_regression(X_train, Y_train)

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------
    rows = []

    for lf in list_label_files(TEST_ROOT):
        for line_idx, line in enumerate(lf.read_text().splitlines()):
            if not line.strip():
                continue

            rec = parse_yolo_pose_line(line.split())
            k = rec["kpts"]

            # Need 3 inputs visible to predict
            if min(k[KP_CARAPACE][2],
                   k[KP_ROSTRUM][2],
                   k[KP_TAIL][2]) < MIN_VIS:
                continue

            X = np.array([[
                k[KP_CARAPACE][0], k[KP_CARAPACE][1],
                k[KP_ROSTRUM][0],  k[KP_ROSTRUM][1],
                k[KP_TAIL][0],     k[KP_TAIL][1],
            ]])

            pred = predict_linear_regression(B, X)[0]

            has_gt = k[KP_EYES][2] >= MIN_VIS
            dist = (
                np.linalg.norm(pred - np.array([k[KP_EYES][0], k[KP_EYES][1]]))
                if has_gt else np.nan
            )

            rows.append({
                "label_file": str(lf),
                "line_idx": line_idx,
                "cls": rec["cls"],

                "e_x": k[KP_EYES][0],
                "e_y": k[KP_EYES][1],
                "e_v": k[KP_EYES][2],

                "e_pred_x": float(pred[0]),
                "e_pred_y": float(pred[1]),

                "dist": dist,
            })

    df = pd.DataFrame(rows)

    df.to_csv(OUT_CSV, index=False)
    df.to_excel(OUT_XLSX, index=False)

    print(f"[OK] Results saved:")
    print(OUT_CSV)
    print(OUT_XLSX)


if __name__ == "__main__":
    main()
