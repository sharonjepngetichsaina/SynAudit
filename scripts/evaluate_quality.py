"""Batch comparison of a real dataset against its synthetic version.

Produces the same CSV files as before (numeric / categorical / correlation) plus:
  * zero-share columns in the numeric table (catches zero-inflated columns)
  * <dataset>-consistency.csv   education <-> education_num style checks
  * <dataset>-sdmetrics.csv     official SDMetrics Column Shapes / Column Pair Trends
  * <dataset>-summary.json      one-line scores, the baseline for the streaming monitor

Usage (from the project root):
    python scripts/evaluate_quality.py --dataset adult
    python scripts/evaluate_quality.py --dataset credit-default
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quality import metrics as m  # noqa: E402


def compare_datasets(dataset_name, with_sdmetrics=True):
    real_path = ROOT / "data" / "processed" / f"{dataset_name}.csv"
    synthetic_path = ROOT / "data" / "synthetic" / f"{dataset_name}-synthetic.csv"
    metadata_path = ROOT / "data" / "synthetic" / f"{dataset_name}-metadata.json"
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not real_path.exists():
        raise FileNotFoundError(f"Missing real dataset: {real_path}")
    if not synthetic_path.exists():
        raise FileNotFoundError(f"Missing synthetic dataset: {synthetic_path}. Generate it first.")

    real = pd.read_csv(real_path)
    synthetic = pd.read_csv(synthetic_path)

    print("\n=== DATASET SHAPES ===")
    print("Real:", real.shape)
    print("Synthetic:", synthetic.shape)

    if set(real.columns) != set(synthetic.columns):
        raise ValueError("Real and synthetic columns do not match.")
    synthetic = synthetic[real.columns]

    ctypes = m.load_column_types(metadata_path) if metadata_path.exists() else m.infer_column_types(real)
    # Metadata may call a column "ordinal"; the numeric/categorical split must match the data.
    numeric_columns = [c for c in real.select_dtypes(include="number").columns
                       if c in synthetic.select_dtypes(include="number").columns]
    categorical_columns = [c for c in real.columns if c not in numeric_columns]
    ctypes = m.ColumnTypes(numeric_columns, categorical_columns, ctypes.feature_ranges)

    print("\n=== MISSING VALUES ===")
    print(pd.DataFrame({"real_missing": real.isna().sum(),
                        "synthetic_missing": synthetic.isna().sum()}).to_string())

    print("\n=== DUPLICATE ROWS ===")
    print("Real duplicate rows:", int(real.duplicated().sum()))
    print("Synthetic duplicate rows:", int(synthetic.duplicated().sum()))

    # ---- numeric distributions (same columns as before + zero shares)
    rows = []
    smd = m.smd_per_column(real, synthetic, numeric_columns)
    rz, sz = m.zero_shares(real, numeric_columns), m.zero_shares(synthetic, numeric_columns)
    for col in numeric_columns:
        r = pd.to_numeric(real[col], errors="coerce").dropna()
        s = pd.to_numeric(synthetic[col], errors="coerce").dropna()
        if r.empty or s.empty:
            continue
        rows.append({
            "column": col,
            "real_mean": r.mean(), "synthetic_mean": s.mean(),
            "real_std": r.std(), "synthetic_std": s.std(),
            "real_min": r.min(), "synthetic_min": s.min(),
            "real_max": r.max(), "synthetic_max": s.max(),
            "standardized_mean_difference": smd[col],
            "real_zero_share": rz[col], "synthetic_zero_share": sz[col],
        })
    numeric_df = pd.DataFrame(rows)
    print("\n=== NUMERIC DISTRIBUTIONS ===")
    if not numeric_df.empty:
        print(numeric_df.to_string(index=False))
        numeric_df.to_csv(results_dir / f"{dataset_name}-numeric-quality.csv", index=False)

    # ---- categorical distributions
    tvd = m.tvd_per_column(real, synthetic, categorical_columns)
    cat_rows = [{
        "column": col,
        "real_unique_categories": int(real[col].nunique(dropna=False)),
        "synthetic_unique_categories": int(synthetic[col].nunique(dropna=False)),
        "total_variation_distance": tvd[col],
    } for col in categorical_columns]
    categorical_df = pd.DataFrame(cat_rows)
    print("\n=== CATEGORICAL DISTRIBUTIONS ===")
    if not categorical_df.empty:
        print(categorical_df.to_string(index=False))
        categorical_df.to_csv(results_dir / f"{dataset_name}-categorical-quality.csv", index=False)

    # ---- correlation structure
    diff = m.corr_difference(real, synthetic, numeric_columns)
    mean_corr = float("nan")
    if diff is not None:
        mean_corr = float(np.nanmean(diff.to_numpy()))
        print("\n=== CORRELATION STRUCTURE ===")
        print("Mean absolute correlation difference:", mean_corr)
        diff.to_csv(results_dir / f"{dataset_name}-correlation-difference.csv")

    # ---- consistency between columns that must agree
    consistency = {}
    for a, b in m.CONSISTENCY_PAIRS.get(dataset_name, []):
        if a in real and b in real:
            valid = m.learn_valid_pairs(real, a, b)
            consistency[f"{a}->{b}"] = m.consistency_rate(synthetic, valid, a, b)
    if consistency:
        print("\n=== CONSISTENCY (share of synthetic rows with a pair that exists in real data) ===")
        for name, rate in consistency.items():
            print(f"{name}: {rate:.4f}")
        pd.DataFrame([{"pair": k, "consistency_rate": v} for k, v in consistency.items()]).to_csv(
            results_dir / f"{dataset_name}-consistency.csv", index=False)

    # ---- zero-inflation gap
    zero_gap = m.zero_share_gap(real, synthetic, numeric_columns)
    if zero_gap:
        print("\n=== ZERO-INFLATED COLUMNS (gap in share of zeros, 0 is perfect) ===")
        for col, gap in zero_gap.items():
            print(f"{col}: real {rz[col]:.3f} vs synthetic {sz[col]:.3f} -> gap {gap:.3f}")

    # ---- SDMetrics (official scores)
    sd = {}
    if with_sdmetrics:
        print("\n=== SDMETRICS QUALITY REPORT ===")
        sd = m.sdmetrics_scores(m.coerce_frame(real, ctypes), m.coerce_frame(synthetic, ctypes), ctypes)
        for k, v in sd.items():
            print(f"{k}: {v:.4f}")
        pd.DataFrame([sd]).to_csv(results_dir / f"{dataset_name}-sdmetrics.csv", index=False)

    summary = {
        "dataset": dataset_name,
        "rows_real": len(real), "rows_synthetic": len(synthetic),
        "smd_mean": float(np.nanmean(list(smd.values()))) if smd else None,
        "smd_max": float(np.nanmax(list(smd.values()))) if smd else None,
        "tvd_mean": float(np.mean(list(tvd.values()))) if tvd else None,
        "corr_diff_mean": mean_corr,
        "zero_gap_max": max(zero_gap.values()) if zero_gap else None,
        "consistency": consistency,
        **sd,
    }
    (results_dir / f"{dataset_name}-summary.json").write_text(
        json.dumps(summary, indent=2, default=lambda o: None if o != o else float(o)), encoding="utf-8")

    print("\nResults saved in:", results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["adult", "credit-default"], required=True)
    parser.add_argument("--no-sdmetrics", action="store_true", help="skip the SDMetrics report (faster)")
    args = parser.parse_args()
    compare_datasets(args.dataset, with_sdmetrics=not args.no_sdmetrics)
