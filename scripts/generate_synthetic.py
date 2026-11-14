"""Generate a synthetic dataset with SDV (GaussianCopula).

Compared with the first version this script fixes several quality problems that
the default model cannot handle on its own. Each fix is configured per dataset,
so the generator stays usable on both datasets without hard-coded column names.

Adult:
1. Zero-inflated columns (capital_gain / capital_loss are ~92% / ~95% zeros).
   A Gaussian copula smears the spike at 0 into a continuous distribution, so the
   synthetic mean was 35,784 instead of 1,079. Fix: treat 0 as "missing" while
   fitting, so SDV models the zero/non-zero decision explicitly, then turn the
   missing values back into 0 after sampling.
2. Deterministic column pairs (education <-> education_num). The model treats them
   as independent, so it can output "HS-grad" with education_num 14. Fix: learn
   education -> education_num from the real data, drop education_num before
   fitting, and derive it from education after sampling.
3. Invalid categorical combinations (relationship / sex / marital_status). The
   copula reproduces each column's marginal shares but links columns through one
   correlation matrix, so it generates impossible rows such as a female "Husband".
   Fix: an SDV FixedCombinations constraint, so only combinations seen in the real
   data can be sampled.

Credit Default:
4. Integer status codes (pay_0, pay_2..pay_6). These are repayment-status codes
   with a large spike at 0 ("paid on time"), which the copula under-produces when
   it treats them as continuous numbers. Fix: model them as categorical in the fit
   metadata, so each code's frequency is matched directly. The saved metadata
   keeps them numeric, so downstream services are unaffected.
5. Heavy-tailed money columns (bill_amt*, pay_amt*).
   - bill_amt* are roughly symmetric but over-spread under the default marginal.
     Fix: a 'truncnorm' marginal, bounded to the real min/max.
   - pay_amt* are a spike of zeros (~20%) plus a heavy positive tail. No single
     distribution captures both, so they are handled with the same zero-inflation
     path as capital_gain: mask the zeros while fitting, model the non-zero
     payments, restore the zeros after sampling. They sit below the automatic
     50%-zeros threshold, so they are listed explicitly in FORCE_ZERO_INFLATED.
   (Two earlier attempts were dropped: a signed-log transform, whose convex inverse
   amplified small tail errors into huge ones; and a 'gaussian_kde' marginal, which
   needs an N x N matrix and runs out of memory at this data size. A 'gamma'
   marginal on pay_amt* collapsed to all zeros, because gamma cannot fit a zero
   spike and a heavy tail at once -- hence the zero-inflation path instead.)

Usage (from the project root):
    python scripts/generate_synthetic.py --dataset adult --num-rows 40000 --seed 42
    python scripts/generate_synthetic.py --dataset credit-default --num-rows 40000 --seed 42
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

# Columns whose joint combinations must exist in the real data
FIXED_COMBINATIONS = {
    "adult": [["relationship", "sex", "marital_status"]],
    "credit-default": [],
}

# Integer-coded columns to model as categorical, so each code's frequency
# (including the spike at 0) is matched instead of smeared by a continuous fit.
# Applied to the fit metadata only; the saved metadata keeps them numeric.
CATEGORICAL_CODES = {
    "adult": [],
    "credit-default": ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"],
}

# Columns to force through the zero-inflation path even though they fall below the
# automatic 50%-zeros threshold: a spike of zeros plus a heavy positive tail, which
# no single marginal distribution can fit.
FORCE_ZERO_INFLATED = {
    "adult": [],
    "credit-default": ["pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6"],
}

# Per-column marginal distribution for heavy-tailed numeric columns, matched to the
# column's shape. truncnorm (bounded to the real range) for the roughly symmetric
# bill_amt columns. All are analytic, unlike gaussian_kde, which needs an N x N
# matrix and runs out of memory at this data size.
NUMERICAL_DISTRIBUTIONS = {
    "adult": {},
    "credit-default": {
        **{f"bill_amt{i}": "truncnorm" for i in range(1, 7)},
    },
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
        out[col] = np.round(pd.to_numeric(out[col], errors="coerce")).astype("int64")
    return out


def quality_check(real_data, synthetic_data, derived, zero_inflated, fixed_combinations=(),
                  categorical_codes=(), dist_columns=()):
    """Print the things this script is supposed to fix."""
    print("\nSelf-check")
    for col in zero_inflated:
        rz, sz = (real_data[col] == 0).mean(), (synthetic_data[col] == 0).mean()
        print(f"  {col}: zeros real {rz:.1%} vs synthetic {sz:.1%} | "
              f"mean real {real_data[col].mean():,.1f} vs synthetic {synthetic_data[col].mean():,.1f}")
    for dst, (src, mapping) in derived.items():
        ok = (synthetic_data[src].map(mapping) == synthetic_data[dst]).mean()
        print(f"  {src} -> {dst} consistent in {ok:.1%} of synthetic rows")
    for cols in fixed_combinations:
        if all(c in real_data.columns for c in cols) and all(c in synthetic_data.columns for c in cols):
            valid = set(map(tuple, real_data[cols].dropna().to_numpy()))
            syn = synthetic_data[cols].dropna()
            ok = syn.apply(lambda r: tuple(r) in valid, axis=1).mean()
            print(f"  {' + '.join(cols)} combination exists in real data for {ok:.1%} of synthetic rows")
    for col in categorical_codes:
        if col in real_data.columns and col in synthetic_data.columns:
            rz, sz = (real_data[col] == 0).mean(), (synthetic_data[col] == 0).mean()
            print(f"  {col}: zero-code share real {rz:.1%} vs synthetic {sz:.1%}")
    for col in dist_columns:
        if col in real_data.columns and col in synthetic_data.columns:
            print(f"  {col}: mean real {real_data[col].mean():,.0f} vs synthetic {synthetic_data[col].mean():,.0f} | "
                  f"std real {real_data[col].std():,.0f} vs synthetic {synthetic_data[col].std():,.0f}")


def generate_synthetic_dataset(dataset_name, num_rows, seed=None, append=False):
    # SDV is imported here so the helper functions above can be tested without it.
    from sdv.metadata import Metadata
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.cag import FixedCombinations

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

    # The saved metadata describes the FINAL schema (all original columns) with the
    # original column types, because the quality service and drift read column
    # types and ranges from it. The categorical-code override below is applied only
    # to the fit metadata, so downstream services still see those columns as numeric.
    output_dir.mkdir(parents=True, exist_ok=True)
    final_metadata = Metadata.detect_from_dataframes(data={dataset_name: real_data})
    metadata_path = output_dir / f"{dataset_name}-metadata.json"
    if metadata_path.exists():
       metadata_path.unlink()
    final_metadata.save_to_json(filepath=str(metadata_path))

    print("Analysing the data...")
    derived = learn_derived_mappings(real_data, dataset_name)
    zero_inflated = find_zero_inflated(real_data, skip=set(derived))
    for col in FORCE_ZERO_INFLATED.get(dataset_name, []):
        if col in real_data.columns and col not in zero_inflated and col not in derived:
            share = float((real_data[col] == 0).mean())
            print(f"  [zero-inflated:forced] {col}: {share:.1%} zeros")
            zero_inflated.append(col)
    categorical_codes = [c for c in CATEGORICAL_CODES.get(dataset_name, []) if c in real_data.columns]

    train_data = prepare_for_fit(real_data, derived, zero_inflated)
    fit_metadata = Metadata.detect_from_dataframes(data={dataset_name: train_data})
    for col in categorical_codes:
        print(f"  [categorical] {col}")
        fit_metadata.update_column(column_name=col, table_name=dataset_name, sdtype="categorical")

    # Per-column marginal for heavy-tailed columns that are still numeric in the fit
    # metadata (categorical overrides take precedence and are excluded).
    numerical_distributions = {
        c: d for c, d in NUMERICAL_DISTRIBUTIONS.get(dataset_name, {}).items()
        if c in train_data.columns and c not in categorical_codes and c not in zero_inflated
    }
    for col, dist in numerical_distributions.items():
        print(f"  [marginal] {col} -> {dist}")

    if seed is not None:
        np.random.seed(seed)

    print("Fitting GaussianCopulaSynthesizer...")
    synthesizer = GaussianCopulaSynthesizer(
        fit_metadata,
        numerical_distributions=numerical_distributions or None,
    )
    for cols in FIXED_COMBINATIONS.get(dataset_name, []):
        print(f"  [fixed combinations] {cols}")
        synthesizer.add_constraints([FixedCombinations(column_names=cols)])
    synthesizer.fit(data={dataset_name: train_data})

    print(f"Generating {num_rows:,} unique synthetic rows...")
    synthetic_data = pd.DataFrame(columns=real_data.columns)
    while len(synthetic_data) < num_rows:
        need = num_rows - len(synthetic_data)
        batch = synthesizer.sample(num_rows=int(need * 1.1) + 10)
        # SDV 2.0 can return a dictionary keyed by table name.
        if isinstance(batch, dict):
            batch = batch[dataset_name] if dataset_name in batch else next(iter(batch.values()))
        batch = restore_after_sampling(batch, real_data, derived, zero_inflated)
        synthetic_data = pd.concat([synthetic_data, batch], ignore_index=True).drop_duplicates(ignore_index=True)
    synthetic_data = synthetic_data.head(num_rows)
    print(f"  unique synthetic rows: {len(synthetic_data):,}")

    if set(synthetic_data.columns) != set(real_data.columns):
        raise ValueError("Generated columns do not match real data columns.")

    if append and output_path.exists():
        existing_data = pd.read_csv(output_path)
        if list(existing_data.columns) != list(real_data.columns):
            raise ValueError(
                f"Existing synthetic schema in {output_path} does not match the processed dataset."
            )
        synthetic_data = pd.concat([existing_data, synthetic_data], ignore_index=True)

    # Write to a sibling temporary file first so an interrupted write cannot
    # leave the existing synthetic dataset truncated.
    temporary_path = output_path.with_suffix(".csv.tmp")
    synthetic_data.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)

    print("\nGeneration successful!")
    print(f"Real data shape: {real_data.shape}")
    print(f"Synthetic data shape: {synthetic_data.shape}")
    print(f"CSV saved to: {output_path}")
    print(f"Metadata saved to: {metadata_path}")
    quality_check(real_data, synthetic_data, derived, zero_inflated,
                  FIXED_COMBINATIONS.get(dataset_name, []), categorical_codes,
                  list(numerical_distributions))


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data using SDV.")
    parser.add_argument("--dataset", choices=["adult", "credit-default"], required=True)
    parser.add_argument("--num-rows", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=None, help="for reproducible output")
    parser.add_argument(
        "--append",
        action="store_true",
        help="append generated rows to the existing synthetic CSV instead of replacing it",
    )
    args = parser.parse_args()

    if args.num_rows <= 0:
        parser.error("--num-rows must be greater than zero")

    generate_synthetic_dataset(args.dataset, args.num_rows, args.seed, args.append)


if __name__ == "__main__":
    main()