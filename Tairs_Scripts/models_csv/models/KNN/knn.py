# language: python
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ============================================================
# CONFIG
# ============================================================
# Current file: Tairs_Scripts/models_csv/models/KNN/knn.py
SCRIPT_DIR = Path(__file__).resolve().parent

# Navigate up to Project Root
# SCRIPT_DIR (KNN) -> models -> models_csv -> Tairs_Scripts -> Project Root
PROJECT_ROOT = SCRIPT_DIR.parents[3]

TRAIN_ROOT = PROJECT_ROOT / "train_on_all"
TEST_ROOT  = PROJECT_ROOT / "prawn_2025_circ_small_v1"

# Output files in the KNN folder
OUT_CSV  = SCRIPT_DIR / "knn_eyes_predictions.csv"
OUT_XLSX = SCRIPT_DIR / "knn_eyes_predictions.xlsx"

# Keypoint indices
KP_CARAPACE = 0
KP_EYES     = 1
KP_ROSTRUM  = 2
KP_TAIL     = 3

NUM_KPTS = 4
MIN_VIS = 1  # use keypoint only if visibility >= 1

# KNN Hyperparameters
N_NEIGHBORS = 5


# ============================================================
# Helpers (Reused from Baseline logic)
# ============================================================
def list_label_files(root: Path) -> list[Path]:
    labels_dir = root / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing labels/ under: {root}")

    files = sorted([p for p in labels_dir.rglob("*.txt") if p.is_file() and p.name != "labels.cache"])
    if not files:
        raise FileNotFoundError(f"No label .txt files found under: {labels_dir}")
    return files


def parse_yolo_pose_line(tokens: list[str], num_kpts: int) -> dict:
    """
    YOLO pose label format per instance:
      class x y w h  x1 y1 v1  x2 y2 v2 ... xK yK vK
    """
    expected = 1 + 4 + 3 * num_kpts
    if len(tokens) < expected:
        raise ValueError(f"Bad label line: got {len(tokens)} tokens.")

    cls = int(float(tokens[0]))
    box = list(map(float, tokens[1:5]))  # x y w h

    kpt_tokens = list(map(float, tokens[5: 5 + 3 * num_kpts]))
    kpts = []
    for i in range(num_kpts):
        x = kpt_tokens[3 * i + 0]
        y = kpt_tokens[3 * i + 1]
        v = int(round(kpt_tokens[3 * i + 2]))
        kpts.append((x, y, v))

    return {"cls": cls, "box": box, "kpts": kpts}


def iter_instances(label_file: Path, num_kpts: int):
    lines = label_file.read_text(encoding="utf-8").splitlines()
    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        rec = parse_yolo_pose_line(tokens, num_kpts=num_kpts)
        rec["label_file"] = str(label_file)
        rec["line_idx"] = line_idx
        yield rec


def main() -> None:
    print("=== KNN Model: Eyes Prediction ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Train root  : {TRAIN_ROOT}")
    print(f"Test root   : {TEST_ROOT}")

    # ------------------------------------------------------------
    # 1) Build TRAIN set
    # ------------------------------------------------------------
    print("Loading training data...")
    train_files = list_label_files(TRAIN_ROOT)
    X_train = []
    Y_train = []

    for lf in train_files:
        for inst in iter_instances(lf, NUM_KPTS):
            (cx, cy, cv) = inst["kpts"][KP_CARAPACE]
            (ex, ey, ev) = inst["kpts"][KP_EYES]
            (rx, ry, rv) = inst["kpts"][KP_ROSTRUM]
            (tx, ty, tv) = inst["kpts"][KP_TAIL]

            # Require all 4 keypoints visible enough for TRAIN
            if min(cv, rv, tv, ev) < MIN_VIS:
                continue

            X_train.append([cx, cy, rx, ry, tx, ty])
            Y_train.append([ex, ey])

    if len(X_train) < 50:
        raise RuntimeError(f"Not enough training instances (found {len(X_train)}).")

    X_train = np.array(X_train, dtype=float)
    Y_train = np.array(Y_train, dtype=float)
    print(f"Training samples: {len(X_train)}")

    # ------------------------------------------------------------
    # 2) Train KNN Model
    # ------------------------------------------------------------
    print(f"Training KNN (k={N_NEIGHBORS})...")
    # Pipeline: Scale features -> KNN
    model = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=N_NEIGHBORS)
    )
    model.fit(X_train, Y_train)

    # ------------------------------------------------------------
    # 3) Predict on TEST set
    # ------------------------------------------------------------
    print("Predicting on test data...")
    test_files = list_label_files(TEST_ROOT)

    rows = []

    for lf in test_files:
        for inst in iter_instances(lf, NUM_KPTS):
            (cx, cy, cv) = inst["kpts"][KP_CARAPACE]
            (ex, ey, ev) = inst["kpts"][KP_EYES]
            (rx, ry, rv) = inst["kpts"][KP_ROSTRUM]
            (tx, ty, tv) = inst["kpts"][KP_TAIL]

            # Need inputs (Carapace, Rostrum, Tail) visible to predict
            if min(cv, rv, tv) < MIN_VIS:
                continue

            # Prepare input vector
            X_in = np.array([[cx, cy, rx, ry, tx, ty]], dtype=float)

            # Predict
            pred = model.predict(X_in)[0]
            e_pred_x, e_pred_y = float(pred[0]), float(pred[1])

            # Calculate error if GT is available
            has_gt_eyes = (ev >= MIN_VIS)
            abs_err_x = abs(ex - e_pred_x) if has_gt_eyes else np.nan
            abs_err_y = abs(ey - e_pred_y) if has_gt_eyes else np.nan
            dist = float(np.sqrt((ex - e_pred_x) ** 2 + (ey - e_pred_y) ** 2)) if has_gt_eyes else np.nan

            rows.append({
                "split": "test",
                "label_file": inst["label_file"],
                "line_idx": inst["line_idx"],
                "cls": inst["cls"],
                "box_x": inst["box"][0],
                "box_y": inst["box"][1],
                "box_w": inst["box"][2],
                "box_h": inst["box"][3],

                # Inputs
                "c_x": cx, "c_y": cy, "c_v": cv,
                "r_x": rx, "r_y": ry, "r_v": rv,
                "t_x": tx, "t_y": ty, "t_v": tv,

                # GT eyes
                "e_x": ex, "e_y": ey, "e_v": ev,

                # Prediction
                "e_pred_x": e_pred_x,
                "e_pred_y": e_pred_y,

                # Errors
                "abs_err_x": abs_err_x,
                "abs_err_y": abs_err_y,
                "dist": dist,
            })

    if not rows:
        raise RuntimeError("No test instances found with visible inputs.")

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------
    # 4) Save Output
    # ------------------------------------------------------------
    df.to_csv(OUT_CSV, index=False)
    df.to_excel(OUT_XLSX, index=False)

    # Quick console summary
    eval_df = df[df["e_v"] >= MIN_VIS]
    if not eval_df.empty:
        mae_x = eval_df["abs_err_x"].mean()
        mae_y = eval_df["abs_err_y"].mean()
        dist_avg = eval_df["dist"].mean()
        print("\n--- Test Metrics (GT eyes visible) ---")
        print(f"MAE_x    : {mae_x:.6f}")
        print(f"MAE_y    : {mae_y:.6f}")
        print(f"Dist_avg : {dist_avg:.6f}")

    print(f"\nSaved results to:\n{OUT_CSV}\n{OUT_XLSX}")


if __name__ == "__main__":
    main()