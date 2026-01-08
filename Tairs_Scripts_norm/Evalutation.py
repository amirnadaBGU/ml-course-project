from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

try:
    # Prefer package import when run as module
    from Tairs_Scripts_norm.config import OUTPUT_BASE
except Exception:
    # Fallback when executed directly: ensure project root on sys.path then import if available
    import sys
    from pathlib import Path
    PROJ_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJ_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJ_ROOT))
    try:
        from Tairs_Scripts_norm.config import OUTPUT_BASE
    except Exception:
        # As a last resort set OUTPUT_BASE to the conventional Tairs_Scripts models folder if it exists
        OUTPUT_BASE = PROJ_ROOT / "Tairs_Scripts" / "models_csv" / "models"

# Evaluate only if GT eyes visibility >= this
MIN_EYES_VIS_FOR_EVAL = 1

# Accuracy thresholds in normalized distance units (body-frame units)
EPS_LIST = [0.005, 0.01, 0.02, 0.05]


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


def compute_metrics(df_eval: pd.DataFrame, label: str, x_col: str = "e_x", y_col: str = "e_y", pred_x: str = "e_pred_x", pred_y: str = "e_pred_y") -> tuple[list[dict], pd.DataFrame]:
    """
    Compute standard regression metrics for eyes prediction.
    Columns are specified so this can work with body-frame outputs (e_l/e_d) or image coords (e_x/e_y).
    """
    for col in [x_col, y_col, pred_x, pred_y]:
        df_eval[col] = pd.to_numeric(df_eval[col], errors="coerce")

    dx = df_eval[x_col] - df_eval[pred_x]
    dy = df_eval[y_col] - df_eval[pred_y]
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

    # Determine which columns exist and map to a standard set
    # Prefer body-frame columns e_l/e_d if present
    if {"e_l", "e_d", "e_pred_l", "e_pred_d"}.issubset(set(df.columns)):
        x_col, y_col, pred_x, pred_y = "e_l", "e_d", "e_pred_l", "e_pred_d"
    elif {"e_x", "e_y", "e_pred_x", "e_pred_y"}.issubset(set(df.columns)):
        x_col, y_col, pred_x, pred_y = "e_x", "e_y", "e_pred_x", "e_pred_y"
    else:
        raise ValueError(f"CSV {csv_path} missing required columns for evaluation.")

    # visibility column (optional); fallback to 1 if not present
    if "e_v" in df.columns:
        df["e_v"] = pd.to_numeric(df["e_v"], errors="coerce")
        df_eval = df[df["e_v"] >= MIN_EYES_VIS_FOR_EVAL].copy()
    else:
        df_eval = df.copy()

    if df_eval.empty:
        print(f"  [Warn] No rows meeting visibility requirement in {csv_path.name}. Skipping.")
        return

    all_rows = []

    # Overall metrics
    overall_metrics, df_eval_with_dist = compute_metrics(df_eval, label="overall", x_col=x_col, y_col=y_col, pred_x=pred_x, pred_y=pred_y)
    all_rows.extend(overall_metrics)

    # By eyes visibility level (e_v) if present
    if "e_v" in df_eval.columns:
        for v in sorted(df_eval["e_v"].dropna().unique().tolist()):
            sub = df_eval[df_eval["e_v"] == v].copy()
            if len(sub) < 10:
                continue
            m, _ = compute_metrics(sub, label=f"eyes_visibility={int(v)}", x_col=x_col, y_col=y_col, pred_x=pred_x, pred_y=pred_y)
            all_rows.extend(m)

    # By class (if exists)
    if "cls" in df_eval.columns:
        df_eval["cls"] = pd.to_numeric(df_eval["cls"], errors="coerce")
        for c in sorted(df_eval["cls"].dropna().unique().tolist()):
            sub = df_eval[df_eval["cls"] == c].copy()
            if len(sub) < 50:
                continue
            m, _ = compute_metrics(sub, label=f"class={int(c)}", x_col=x_col, y_col=y_col, pred_x=pred_x, pred_y=pred_y)
            all_rows.extend(m)

    # By image (label_file) worst 10 if label_file exists
    if "label_file" in df_eval_with_dist.columns:
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
        print(f"  MAE_x    : {row.get('MAE_x', row.get('MAE_l', float('nan'))):.6f}")
        print(f"  MAE_y    : {row.get('MAE_y', row.get('MAE_d', float('nan'))):.6f}")
        print(f"  Dist_avg : {row.get('Dist_avg'):.6f}")

    print(f"  Saved summary to: {out_csv}")


def main() -> None:
    # Search multiple model roots so this evaluator works for both original and normalized scripts
    models_roots = []
    # normalized workspace
    if OUTPUT_BASE is not None:
        models_roots.append(Path(OUTPUT_BASE))
    # also check original Tairs_Scripts models location relative to project root
    proj_root = Path(__file__).resolve().parents[1]
    other_root = proj_root / "Tairs_Scripts" / "models_csv" / "models"
    models_roots.append(other_root)

    # collect CSVs from all existing roots, avoid duplicates
    csv_set = []
    for root in models_roots:
        try:
            root = Path(root)
        except Exception:
            continue
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.csv")):
            if p not in csv_set:
                csv_set.append(p)

    if not csv_set:
        print(f"No model CSVs found in any of: {models_roots}")
        return

    for csv_path in csv_set:
        # Skip training features and existing summary files
        if csv_path.name.startswith("features_") or csv_path.name.startswith("evaluation_summary_"):
            continue

        try:
            evaluate_csv(csv_path)
        except Exception as e:
            print(f"Failed to evaluate {csv_path}: {e}")


if __name__ == "__main__":
    main()
