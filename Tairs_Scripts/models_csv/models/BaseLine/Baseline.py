#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# CONFIG (no terminal needed)
# Place this file under: ml-course-project/Tairs_Scripts/
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent                 # .../ml-course-project/Tairs_Scripts
PROJECT_ROOT = SCRIPT_DIR.parent                             # .../ml-course-project

TRAIN_ROOT = PROJECT_ROOT / "train_on_all"                   # TRAIN dataset root (has labels/)
TEST_ROOT  = PROJECT_ROOT / "prawn_2025_circ_small_v1"        # TEST dataset root (has labels/)

OUT_DIR = SCRIPT_DIR / "models_csv"
OUT_BASENAME = "baseline_linreg_eyes_from3kpts_test"
OUT_CSV  = OUT_DIR / f"{OUT_BASENAME}.csv"
OUT_XLSX = OUT_DIR / f"{OUT_BASENAME}.xlsx"

# --- Keypoint indices (as you specified) ---
KP_CARAPACE = 0
KP_EYES     = 1
KP_ROSTRUM  = 2
KP_TAIL     = 3

NUM_KPTS = 4
MIN_VIS = 1  # use keypoint only if visibility >= 1

# ============================================================
# Helpers
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
    coords normalized [0,1], visibility v in {0,1,2}.
    """
    expected = 1 + 4 + 3 * num_kpts
    if len(tokens) < expected:
        raise ValueError(
            f"Bad label line: got {len(tokens)} tokens, expected >= {expected}. "
            f"First tokens: {tokens[:12]}"
        )

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


def fit_linear_regression(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Fit multivariate linear regression with intercept using least squares.
    X: (N, d), Y: (N, 2)
    Returns B: (d+1, 2) such that Y_hat = [1, X] @ B
    """
    if X.ndim != 2 or Y.ndim != 2 or Y.shape[1] != 2:
        raise ValueError("Shapes must be X=(N,d), Y=(N,2).")

    N = X.shape[0]
    X1 = np.hstack([np.ones((N, 1), dtype=float), X])  # add intercept
    B, *_ = np.linalg.lstsq(X1, Y, rcond=None)
    return B


def predict_linear_regression(B: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    B: (d+1,2), X:(N,d) -> Y_hat:(N,2)
    """
    N = X.shape[0]
    X1 = np.hstack([np.ones((N, 1), dtype=float), X])
    return X1 @ B


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1) Build TRAIN set: use SAME prawn instance
    #    Inputs: (carapace, rostrum, tail) -> Output: eyes
    # ------------------------------------------------------------
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
        raise RuntimeError(
            f"Not enough training instances with all keypoints visible (found {len(X_train)}). "
            "Try lowering MIN_VIS or check labels."
        )

    X_train = np.array(X_train, dtype=float)
    Y_train = np.array(Y_train, dtype=float)

    # Fit baseline linear regression
    B = fit_linear_regression(X_train, Y_train)

    print("=== Baseline Model: Linear Regression (Eyes from 3 keypoints) ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Train root  : {TRAIN_ROOT}")
    print(f"Test root   : {TEST_ROOT}")
    print(f"Saving CSV  : {OUT_CSV}")
    print(f"Saving XLSX : {OUT_XLSX}")
    print(f"Train samples used: {len(X_train)}")
    print("Keypoints: carapace=0, eyes=1, rostrum=2, tail=3")
    print("Model: [1, cx, cy, rx, ry, tx, ty] @ B -> (ex, ey)")

    # ------------------------------------------------------------
    # 2) Predict on TEST per-instance (same prawn instance)
    #    We can predict when (carapace, rostrum, tail) visible.
    #    We evaluate only when eyes also visible.
    # ------------------------------------------------------------
    test_files = list_label_files(TEST_ROOT)

    rows = []
    eval_dists = []

    for lf in test_files:
        for inst in iter_instances(lf, NUM_KPTS):
            (cx, cy, cv) = inst["kpts"][KP_CARAPACE]
            (ex, ey, ev) = inst["kpts"][KP_EYES]
            (rx, ry, rv) = inst["kpts"][KP_ROSTRUM]
            (tx, ty, tv) = inst["kpts"][KP_TAIL]

            # Need the 3 inputs visible to predict
            if min(cv, rv, tv) < MIN_VIS:
                continue

            X = np.array([[cx, cy, rx, ry, tx, ty]], dtype=float)
            pred = predict_linear_regression(B, X)[0]
            e_pred_x, e_pred_y = float(pred[0]), float(pred[1])

            # Error only if GT eyes visible
            has_gt_eyes = (ev >= MIN_VIS)
            abs_err_x = abs(ex - e_pred_x) if has_gt_eyes else np.nan
            abs_err_y = abs(ey - e_pred_y) if has_gt_eyes else np.nan
            dist = float(np.sqrt((ex - e_pred_x) ** 2 + (ey - e_pred_y) ** 2)) if has_gt_eyes else np.nan

            if has_gt_eyes:
                eval_dists.append(dist)

            rows.append({
                "split": "test",
                "label_file": inst["label_file"],
                "line_idx": inst["line_idx"],
                "cls": inst["cls"],
                "box_x": inst["box"][0],
                "box_y": inst["box"][1],
                "box_w": inst["box"][2],
                "box_h": inst["box"][3],

                # Inputs (same prawn)
                "c_x": cx, "c_y": cy, "c_v": cv,
                "r_x": rx, "r_y": ry, "r_v": rv,
                "t_x": tx, "t_y": ty, "t_v": tv,

                # GT eyes (if available)
                "e_x": ex, "e_y": ey, "e_v": ev,

                # Prediction
                "e_pred_x": e_pred_x,
                "e_pred_y": e_pred_y,

                # Errors (NaN if no GT eyes)
                "abs_err_x": abs_err_x,
                "abs_err_y": abs_err_y,
                "dist": dist,
            })

    if not rows:
        raise RuntimeError("No test instances found with visible (carapace, rostrum, tail). Check MIN_VIS/test labels.")

    df = pd.DataFrame(rows)

    # Metrics over rows with GT eyes visible
    eval_df = df[df["e_v"] >= MIN_VIS].copy()
    if len(eval_df) == 0:
        print("\nWARNING: No GT eyes visible in TEST under MIN_VIS, so metrics cannot be computed.")
    else:
        mae_x = float(eval_df["abs_err_x"].mean())
        mae_y = float(eval_df["abs_err_y"].mean())
        dist_avg = float(eval_df["dist"].mean())
        rmse = float(np.sqrt(((eval_df["e_x"] - eval_df["e_pred_x"]) ** 2 + (eval_df["e_y"] - eval_df["e_pred_y"]) ** 2).mean()))

        print("\n--- Test metrics (only where GT eyes visible) ---")
        print(f"Evaluated instances: {len(eval_df)}")
        print(f"MAE_x    : {mae_x:.6f}")
        print(f"MAE_y    : {mae_y:.6f}")
        print(f"RMSE     : {rmse:.6f}")
        print(f"Dist_avg : {dist_avg:.6f}")

    # Save outputs
    df.to_csv(OUT_CSV, index=False)
    df.to_excel(OUT_XLSX, index=False)

    print(f"\nSaved:\n{OUT_CSV}\n{OUT_XLSX}")


if __name__ == "__main__":
    main()
