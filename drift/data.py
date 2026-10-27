"""Load the team's datasets for drift detection.

Real data:      data/processed/{dataset}.csv          (ingestion/preprocess.py)
Synthetic data: data/synthetic/{dataset}-synthetic.csv (synthetic generator)
Column types:   data/synthetic/{dataset}-metadata.json (SDV metadata)

Paths are relative to the repository root, so run scripts from there.

Drift here means the synthetic stream changing compared with its own
earlier behaviour. So the reference is the first part of the synthetic
data, not the real data (the real-vs-synthetic gap is measured by the
quality service).
"""

import json
from pathlib import Path

import pandas as pd

PROCESSED = Path("data/processed")
SYNTHETIC = Path("data/synthetic")

# Size of the reference sample. 4,000 rather than 2,000: with a smaller
# reference, one unusual reference sample makes several windows alarm at
# once (clustered false alarms in the chi-square test, see calibration).
REF_SIZE = 4000

TARGETS = {
    "adult": "income",
    "credit-default": "default_payment_next_month",
}


def load_real(dataset="adult"):
    """Real records in the canonical form (empty cells are missing values)."""
    return pd.read_csv(PROCESSED / f"{dataset}.csv")


def load_synthetic(dataset="adult"):
    """Synthetic records produced by the generator."""
    return pd.read_csv(SYNTHETIC / f"{dataset}-synthetic.csv")


def column_types(dataset="adult", df=None):
    """Return (numeric_columns, categorical_columns), without the target.

    Uses the SDV metadata file when it exists, because it records the type
    the generator itself used. Otherwise falls back to the pandas dtypes.
    """
    target = TARGETS.get(dataset)
    meta_path = SYNTHETIC / f"{dataset}-metadata.json"

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        # single-table metadata has "columns"; multi-table nests it under "tables"
        if "columns" not in meta and "tables" in meta:
            meta = next(iter(meta["tables"].values()))
        sdtypes = {col: info.get("sdtype") for col, info in meta["columns"].items()}
        numeric = [c for c, t in sdtypes.items() if t == "numerical" and c != target]
        categorical = [c for c, t in sdtypes.items()
                       if t in ("categorical", "boolean") and c != target]
        return numeric, categorical

    if df is None:
        raise ValueError("no metadata file found, pass the DataFrame as df=")
    numeric = [c for c in df.select_dtypes(include="number").columns if c != target]
    categorical = [c for c in df.select_dtypes(exclude="number").columns if c != target]
    return numeric, categorical


def reference_and_stream(df, ref_size=REF_SIZE, shuffle_seed=None):
    """Split records into a reference sample and the stream that follows it.

    The first `ref_size` records are the reference: what 'normal' looks like.
    Everything after it is the stream we monitor. If `shuffle_seed` is given,
    the records are shuffled first (with a fixed seed, so the split is
    reproducible); use this to repeat an experiment with different orders.
    """
    if shuffle_seed is not None:
        df = df.sample(frac=1.0, random_state=shuffle_seed)
    df = df.reset_index(drop=True)
    if ref_size >= len(df):
        raise ValueError(f"ref_size={ref_size} leaves no stream ({len(df)} records)")
    reference = df.iloc[:ref_size].reset_index(drop=True)
    stream = df.iloc[ref_size:].reset_index(drop=True)
    return reference, stream


if __name__ == "__main__":
    real = load_real("adult")
    syn = load_synthetic("adult")
    num, cat = column_types("adult", syn)
    ref, stream = reference_and_stream(syn)
    print(f"real:      {real.shape[0]:,} rows x {real.shape[1]} columns")
    print(f"synthetic: {syn.shape[0]:,} rows x {syn.shape[1]} columns")
    print(f"numeric ({len(num)}):     {num}")
    print(f"categorical ({len(cat)}): {cat}")
    print(f"reference: {len(ref):,} records   stream: {len(stream):,} records "
          f"-> {len(stream) // 500} windows of 500")
    missing = syn.isna().sum()
    print("missing values in synthetic data:",
          missing[missing > 0].to_dict() or "none")
