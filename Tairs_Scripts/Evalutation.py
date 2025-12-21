#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# CONFIG (no terminal needed)
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models_csv"

# Default input (the CSV produced by Baseline_Eyes_From3Kpts.py)
IN_CSV = MODELS_DIR / "baseline_linreg_eyes_from3kpts_test.csv"

# Output files
OUT_SUMMARY_XLSX = MODELS_DIR / "evaluation_baseline_summary.xlsx"
OUT_SUMMARY_CSV  = MODELS_DIR / "evaluation_summary.csv"

# Evaluate only if GT eyes visibility >= this
MIN_EYES_VIS_FOR_EVAL = 1

# Accuracy thresholds in normalized distance units
# (tune these if you want tighter/looser)
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

def compute_metrics(df_eval: pd.DataFrame, label: str) -> list[dict]:
    """
    Compute standard regression metrics for eyes prediction.
    Expects columns: e_x, e_y, e_pred_x, e_pred_y
    """
    # numeric conversion
    for col in ["e_x", "e_y", "e_pred_x", "e_pred_y"]:
        df_eval[col] = pd.to_numeric(df_eval[col], errors="coerce")

    dx = df_eval["e_x"] - df_eval["e_pred_x"]
    dy = df_eval["e_y"] - df_eval["e_pred_y"]
    dist = np.sqrt((dx ** 2) + (dy ** 2))

    # store dist for later reporting
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

    # Accuracy@eps
    for eps in EPS_LIST:
        acc = float(np.nanmean((dist <= eps).astype(float)))
        out.append({
            "group": label,
            "n": int(df_eval.shape[0]),
            "metric": f"Accuracy@{eps}",
            "value": acc
        })

    return out, df_eval


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found:\n{IN_CSV}\n"
            "Make sure you ran the model script and it saved the CSV to models_csv/."
        )

    df = pd.read_csv(IN_CSV)

    required = {"e_x", "e_y", "e_pred_x", "e_pred_y", "e_v"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    # filter to rows with GT eyes visible enough
    df["e_v"] = pd.to_numeric(df["e_v"], errors="coerce")
    df_eval = df[df["e_v"] >= MIN_EYES_VIS_FOR_EVAL].copy()

    if df_eval.empty:
        raise RuntimeError(
            f"No rows with e_v >= {MIN_EYES_VIS_FOR_EVAL}. "
            "Cannot evaluate without ground-truth eyes."
        )

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

    # By image (label_file) worst 10 (useful debug)
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

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_SUMMARY_CSV, index=False)
    out_df.to_excel(OUT_SUMMARY_XLSX, index=False)

    # Print a short human-friendly summary
    print("=== Evaluation Summary ===")
    print(f"Input CSV: {IN_CSV}")
    print(f"Rows evaluated (e_v>={MIN_EYES_VIS_FOR_EVAL}): {len(df_eval)}")
    overall = out_df[(out_df["group"] == "overall") & (out_df.get("MAE_x").notna())]
    if len(overall):
        row = overall.iloc[0]
        print(f"MAE_x    : {row['MAE_x']:.6f}")
        print(f"MAE_y    : {row['MAE_y']:.6f}")
        print(f"RMSE     : {row['RMSE']:.6f}")
        print(f"Dist_avg : {row['Dist_avg']:.6f}")
        print(f"Dist_p95 : {row['Dist_p95']:.6f}")

    print(f"\nSaved summary CSV : {OUT_SUMMARY_CSV}")
    print(f"Saved summary XLSX: {OUT_SUMMARY_XLSX}")


if __name__ == "__main__":
    main()
