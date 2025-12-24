# language: python
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models_csv"

# Fallback if no models found in subfolder (optional)
IN_CSV_FALLBACK = MODELS_DIR / "baseline_linreg_eyes_from3kpts_test.csv"

# Evaluate only if GT eyes visibility >= this
MIN_EYES_VIS_FOR_EVAL = 1

# Accuracy thresholds in normalized distance units
EPS_LIST = [0.005, 0.01, 0.02, 0.05]

# ============================================================
# Helpers
# ============================================================
def safe_mean(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    return float(np.nanmean(x.values)) if np.isfinite(x.values).any() else float("nan")

def safe_median(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    return float(np.nanmedian(x.values)) if np.isfinite(x.values).any() else float("nan")

def safe_quantile(x: pd.Series, q: float) -> float:
    x = pd.to_numeric(x, errors="coerce")
    vals = x.values.astype(float)
    vals = vals[np.isfinite(vals)]
    return float(np.quantile(vals, q)) if len(vals) else float("nan")

def compute_metrics(df_eval: pd.DataFrame, label: str) -> tuple[list[dict], pd.DataFrame]:
    """
    Compute standard regression metrics for eyes prediction.
    Expects columns: e_x, e_y, e_pred_x, e_pred_y
    """
    for col in ["e_x", "e_y", "e_pred_x", "e_pred_y"]:
        df_eval[col] = pd.to_numeric(df_eval[col], errors="coerce")

    dx = df_eval["e_x"] - df_eval["e_pred_x"]
    dy = df_eval["e_y"] - df_eval["e_pred_y"]
    dist = np.sqrt((dx ** 2) + (dy ** 2))

    df_eval = df_eval.copy()
    df_eval["dist_eval"] = dist

    mae_x = float(np.nanmean(np.abs(dx)))
    mae_y = float(np.nanmean(np.abs(dy)))
    rmse = float(np.sqrt(np.nanmean(dx**2 + dy**2)))
    dist_avg = float(np.nanmean(dist))

    out = []
    out.append({
        "group": label,
        "n": int(df_eval.shape[0]),
        "MAE_x": mae_x,
        "MAE_y": mae_y,
        "RMSE": rmse,
        "Dist_avg": dist_avg,
        "Dist_median": safe_median(pd.Series(dist)),
        "Dist_p90": safe_quantile(pd.Series(dist), 0.90),
        "Dist_p95": safe_quantile(pd.Series(dist), 0.95),
        "Dist_p99": safe_quantile(pd.Series(dist), 0.99),
    })

    for eps in EPS_LIST:
        acc = float(np.nanmean((dist <= eps).astype(float)))
        out.append({
            "group": label,
            "n": int(df_eval.shape[0]),
            "metric": f"Accuracy@{eps}",
            "value": acc
        })

    return out, df_eval


def evaluate_csv(csv_path: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    print(f"Evaluating: {csv_path.name}")
    df = pd.read_csv(csv_path)

    required = {"e_x", "e_y", "e_pred_x", "e_pred_y", "e_v"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV {csv_path}: {sorted(missing)}")

    df["e_v"] = pd.to_numeric(df["e_v"], errors="coerce")
    df_eval = df[df["e_v"] >= MIN_EYES_VIS_FOR_EVAL].copy()

    if df_eval.empty:
        print(f"  [Warn] No rows with e_v >= {MIN_EYES_VIS_FOR_EVAL} in {csv_path.name}. Skipping.")
        return

    all_rows = []

    # Overall metrics
    overall_metrics, df_eval_with_dist = compute_metrics(df_eval, label="overall")
    all_rows.extend(overall_metrics)

    # By eyes visibility level (e_v)
    for v in sorted(df_eval["e_v"].dropna().unique().tolist()):
        sub = df_eval[df_eval["e_v"] == v].copy()
        if len(sub) < 10:
            continue
        m, _ = compute_metrics(sub, label=f"eyes_visibility={int(v)}")
        all_rows.extend(m)

    # By class (if exists)
    if "cls" in df_eval.columns:
        df_eval["cls"] = pd.to_numeric(df_eval["cls"], errors="coerce")
        for c in sorted(df_eval["cls"].dropna().unique().tolist()):
            sub = df_eval[df_eval["cls"] == c].copy()
            if len(sub) < 50:
                continue
            m, _ = compute_metrics(sub, label=f"class={int(c)}")
            all_rows.extend(m)

    # By image (label_file) worst 10
    if "label_file" in df_eval.columns:
        df_tmp = df_eval_with_dist.copy()
        df_tmp["label_file"] = df_tmp["label_file"].astype(str)
        per_img = (
            df_tmp.groupby("label_file", as_index=False)["dist_eval"]
            .mean()
            .sort_values("dist_eval", ascending=False)
            .head(10)
        )
        for _, r in per_img.iterrows():
            all_rows.append({
                "group": "worst_images_meanDist",
                "label_file": r["label_file"],
                "mean_dist": float(r["dist_eval"]),
            })

    out_df = pd.DataFrame(all_rows)

    # Save outputs next to the input CSV (in the specific model folder)
    out_csv = csv_path.parent / f"evaluation_summary_{csv_path.stem}.csv"
    out_xlsx = csv_path.parent / f"evaluation_summary_{csv_path.stem}.xlsx"

    out_df.to_csv(out_csv, index=False)
    out_df.to_excel(out_xlsx, index=False)

    print("  === Evaluation Summary ===")
    print(f"  Rows evaluated: {len(df_eval)}")
    overall = out_df[(out_df["group"] == "overall") & (out_df.get("MAE_x").notna())]
    if len(overall):
        row = overall.iloc[0]
        print(f"  MAE_x    : {row['MAE_x']:.6f}")
        print(f"  MAE_y    : {row['MAE_y']:.6f}")
        print(f"  Dist_avg : {row['Dist_avg']:.6f}")

    print(f"  Saved summary to: {out_csv}")


def main() -> None:
    # Look for model CSVs under MODELS_DIR / "models"
    models_root = MODELS_DIR / "models"
    csv_list = []
    if models_root.exists():
        csv_list = sorted(models_root.rglob("*.csv"))

    # If none found, fall back to IN_CSV_FALLBACK
    if not csv_list:
        if IN_CSV_FALLBACK.exists():
            csv_list = [IN_CSV_FALLBACK]
        else:
            print(f"No model CSVs found in {models_root} and fallback {IN_CSV_FALLBACK} missing.")
            return

    for csv_path in csv_list:
        # Skip training features and existing summary files
        # This ensures we only evaluate prediction outputs (like knn_eyes_predictions.csv)
        if csv_path.name.startswith("features_") or csv_path.name.startswith("evaluation_summary_"):
            continue

        try:
            evaluate_csv(csv_path)
        except Exception as e:
            print(f"Failed to evaluate {csv_path}: {e}")


if __name__ == "__main__":
    main()