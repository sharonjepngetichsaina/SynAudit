"""Load result files the services already wrote into InfluxDB, for the dashboard.

Works out what each file is from its columns:
  Sharon's utility CSV  (results/utility/*-utility.csv)   -> utility
  Sharon's privacy CSV  (results/privacy/*-copies.csv)    -> privacy
  Grace's drift rows    (CSV with window_id, column, detector, p_value, alarm) -> drift_events
  Matilda's DGIM snapshot (JSON with dataset, source, rates)                -> dgim

    python scripts/import_results.py results/utility/*.csv results/privacy/*.csv
    python scripts/import_results.py --run-id E1-baseline path/to/dgim-snapshot.json
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from storage import db  # noqa: E402


def load(path: Path, run_id):
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        return "dgim", db.write_dgim_snapshot(payload, run_id=run_id)
    df = pd.read_csv(path)
    cols = set(df.columns)
    if {"bal_acc_synthetic", "bal_acc_real"} <= cols:
        return "utility", db.write_utility_rows(df, run_id=run_id)
    if "exact_copy_rate" in cols:
        return "privacy", db.write_privacy_rows(df, run_id=run_id)
    if {"window_id", "column", "detector", "p_value"} <= cols:
        return "drift_events", db.write_drift_rows(df, run_id=run_id)
    return None, 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--run-id", default=None, help="experiment label shown on the dashboard")
    args = ap.parse_args()
    for path in args.files:
        kind, n = load(path, args.run_id)
        print(f"{path}: {'skipped (unknown format)' if kind is None else f'{n} rows -> {kind}'}")


if __name__ == "__main__":
    main()
