"""Generate a synthetic dataset with SDV (GaussianCopula).

Compared with the first version this script fixes two quality problems that the
default model cannot handle on its own:

1. Zero-inflated columns (Adult capital_gain / capital_loss are ~92% / ~95% zeros).
   A Gaussian copula smears the spike at 0 into a continuous distribution, so the
   synthetic mean was 35,784 instead of 1,079. Fix: treat 0 as "missing" while
   fitting, so SDV models the zero/non-zero decision explicitly (as a null
   indicator correlated with the other columns), then turn the missing values
   back into 0 after sampling.

2. Deterministic column pairs (Adult education <-> education_num). The model treats
   them as independent columns, so it can output "HS-grad" with education_num 14.
   Fix: learn the mapping education -> education_num from the real data, drop
   education_num before fitting, and derive it from education after sampling.

Usage (from the project root):
    python scripts/generate_synthetic.py --dataset adult --num-rows 10000
    python scripts/generate_synthetic.py --dataset credit-default --num-rows 10000
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# (source categorical column, derived numeric column)
DERIVED_COLUMNS = {
    "adult": [("education", "education_num")],
    "credit-default": [],
}

ZERO_INFLATED_MIN_SHARE = 0.5   # numeric column with >= 50% zeros is zero-inflated
MIN_DISTINCT_NONZERO = 20       # ...and have >= 20 distinct non-zero values (so 0/1 flags and
                                # category codes like pay_5 are NOT treated as zero-inflated)
MIN_MAPPING_FIDELITY = 0.99     # derived column must follow its source in >= 99% of rows


def learn_derived_mappings(real_data, dataset_name):
    """Return {derived_col: (source_col, {source_value: derived_value})}."""
    mappings = {}
    for src, dst in DERIVED_COLUMNS.get(dataset_name, []):
        if src not in real_data or dst not in real_data:
            continue
        pairs = real_data[[src, dst]].dropna()
        mapping = pairs.groupby(src)[dst].agg(lambda s: s.mode().iloc[0]).to_dict()
        fidelity = float((pairs[src].map(mapping) == pairs[dst]).mean())
        if fidelity < MIN_MAPPING_FIDELITY:
            print(f"  [skip] {dst} does not follow {src} (fidelity {fidelity:.3f})")
            continue
        print(f"  [derived] {dst} <- {src} ({len(mapping)} values, fidelity {fidelity:.3f})")
        mappings[dst] = (src, mapping)
    return mappings


def find_zero_inflated(real_data, skip=()):
    """Numeric columns where at least ZERO_INFLATED_MIN_SHARE of the values are 0."""
    cols = []
    for col in real_data.select_dtypes(include="number").columns:
        if col in skip:
            continue
        share = float((real_data[col] == 0).mean())
        distinct = real_data.loc[real_data[col] != 0, col].nunique()
        if share >= ZERO_INFLATED_MIN_SHARE and distinct >= MIN_DISTINCT_NONZERO:
            print(f"  [zero-inflated] {col}: {share:.1%} zeros")
            cols.append(col)
    return cols


def prepare_for_fit(real_data, derived, zero_inflated):
    """Training frame: derived columns dropped, zeros in zero-inflated columns -> NaN."""
    train = real_data.drop(columns=list(derived)).copy()
    for col in zero_inflated:
        train[col] = train[col].astype("float64").mask(train[col] == 0)
    return train


def restore_after_sampling(sampled, real_data, derived, zero_inflated):
    """Undo prepare_for_fit and return a frame with the original columns and dtypes."""
    out = sampled.copy()
    for col in zero_inflated:
        out[col] = out[col].fillna(0)
    for dst, (src, mapping) in derived.items():
        out[dst] = out[src].map(mapping)
    out = out[real_data.columns]
    for col in real_data.select_dtypes(include="integer").columns:
        out[col] = np.round(out[col]).astype("int64")
    return out


def quality_check(real_data, synthetic_data, derived, zero_inflated):
    """Print the two things this script is supposed to fix."""
    print("\nSelf-check")
    for col in zero_inflated:
        rz, sz = (real_data[col] == 0).mean(), (synthetic_data[col] == 0).mean()
        print(f"  {col}: zeros real {rz:.1%} vs synthetic {sz:.1%} | "
              f"mean real {real_data[col].mean():,.1f} vs synthetic {synthetic_data[col].mean():,.1f}")
    for dst, (src, mapping) in derived.items():
        ok = (synthetic_data[src].map(mapping) == synthetic_data[dst]).mean()
        print(f"  {src} -> {dst} consistent in {ok:.1%} of synthetic rows")


def generate_synthetic_dataset(dataset_name, num_rows, seed=None):
    # SDV is imported here so the helper functions above can be tested without it.
    from sdv.metadata import Metadata
    from sdv.single_table import GaussianCopulaSynthesizer

    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "data" / "processed" / f"{dataset_name}.csv"
    output_dir = project_root / "data" / "synthetic"
    output_path = output_dir / f"{dataset_name}-synthetic.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    real_data = pd.read_csv(input_path)
    if real_data.empty:
        raise ValueError("The real dataset is empty.")
    if real_data.columns.duplicated().any():
        raise ValueError("Duplicate column names detected.")

    print(f"\nDataset: {dataset_name}")
    print(f"Real data shape: {real_data.shape}")

    # The saved metadata describes the FINAL schema (all original columns), because
    # the quality service reads column types and ranges from it.
    output_dir.mkdir(parents=True, exist_ok=True)
    final_metadata = Metadata.detect_from_dataframes(data={dataset_name: real_data})
    metadata_path = output_dir / f"{dataset_name}-metadata.json"
    if metadata_path.exists():
       metadata_path.unlink()
    final_metadata.save_to_json(filepath=str(metadata_path))

    print("Analysing the data...")
    derived = learn_derived_mappings(real_data, dataset_name)
    zero_inflated = find_zero_inflated(real_data, skip=set(derived))

    train_data = prepare_for_fit(real_data, derived, zero_inflated)
    fit_metadata = Metadata.detect_from_dataframes(data={dataset_name: train_data})

    if seed is not None:
        np.random.seed(seed)

    print("Fitting GaussianCopulaSynthesizer...")
    synthesizer = GaussianCopulaSynthesizer(fit_metadata)
    synthesizer.fit(data={dataset_name: train_data})

    print(f"Generating {num_rows:,} synthetic rows...")
    sampled = synthesizer.sample(num_rows=num_rows)

    # SDV 2.0 can return a dictionary keyed by table name.
    if isinstance(sampled, dict):
        if dataset_name in sampled:
            sampled = sampled[dataset_name]
        elif len(sampled) == 1:
            sampled = next(iter(sampled.values()))
        else:
            raise ValueError(f"Unexpected sampled tables: {list(sampled.keys())}")
    if not isinstance(sampled, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(sampled).__name__}")

    synthetic_data = restore_after_sampling(sampled, real_data, derived, zero_inflated)

    if set(synthetic_data.columns) != set(real_data.columns):
        raise ValueError("Generated columns do not match real data columns.")

    synthetic_data.to_csv(output_path, index=False)

    print("\nGeneration successful!")
    print(f"Real data shape: {real_data.shape}")
    print(f"Synthetic data shape: {synthetic_data.shape}")
    print(f"CSV saved to: {output_path}")
    print(f"Metadata saved to: {metadata_path}")
    quality_check(real_data, synthetic_data, derived, zero_inflated)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data using SDV.")
    parser.add_argument("--dataset", choices=["adult", "credit-default"], required=True)
    parser.add_argument("--num-rows", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=None, help="for reproducible output")
    args = parser.parse_args()

    if args.num_rows <= 0:
        parser.error("--num-rows must be greater than zero")

    generate_synthetic_dataset(args.dataset, args.num_rows, args.seed)


if __name__ == "__main__":
    main()
