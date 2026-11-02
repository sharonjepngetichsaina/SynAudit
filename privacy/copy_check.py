"""Exact-copy privacy check with a Bloom filter.

Adds every real row to a Bloom filter, then streams the synthetic rows and
counts how many look like exact copies of a real row, per window.
As a baseline it also measures how often real rows repeat each other.
"""

import argparse
from pathlib import Path

import pandas as pd

from bloom import BloomFilter


def fingerprint(row, columns):
    """Turn one row into a fixed piece of text, the same way every time."""
    parts = []
    for col in columns:
        value = row[col]
        if pd.isna(value):
            parts.append("")
        elif isinstance(value, float) and value.is_integer():
            parts.append(str(int(value)))
        else:
            parts.append(str(value).strip())
    return "|".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Bloom filter exact-copy check.")
    parser.add_argument("--dataset", choices=["adult", "credit-default"], default="adult")
    parser.add_argument("--window", type=int, default=500)
    args = parser.parse_args()

    real = pd.read_csv(Path("data/processed") / f"{args.dataset}.csv")
    synthetic = pd.read_csv(Path("data/synthetic") / f"{args.dataset}-synthetic.csv")
    columns = list(real.columns)

    # Baseline: stream the real rows, check each one before adding it.
    bloom = BloomFilter(expected_items=len(real), false_positive_rate=0.01)
    real_repeats = 0
    for _, row in real.iterrows():
        fp = fingerprint(row, columns)
        if fp in bloom:
            real_repeats += 1
        bloom.add(fp)
    print(f"real rows that repeat an earlier real row: {real_repeats / len(real):.3f}")

    # Exact set of real fingerprints, only to check the Bloom filter's answers.
    exact = {fingerprint(row, columns) for _, row in real.iterrows()}

    results = []
    hits, exact_hits, seen = 0, 0, 0
    for _, row in synthetic.iterrows():
        fp = fingerprint(row, columns)
        seen += 1
        hits += fp in bloom
        exact_hits += fp in exact
        if seen % args.window == 0:
            results.append({
                "dataset": args.dataset,
                "window": seen // args.window,
                "rows_seen": seen,
                "bloom_copy_rate": hits / seen,
                "exact_copy_rate": exact_hits / seen,
            })
            print(f"rows {seen}: Bloom copy rate {hits / seen:.4f}  exact {exact_hits / seen:.4f}")

    out_dir = Path("results/privacy")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}-copies.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Saved {len(results)} windows to {out_path}")


if __name__ == "__main__":
    main()