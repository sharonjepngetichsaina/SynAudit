"""Write fake results to InfluxDB so the dashboard can be shown before the
evaluators are connected. Drift starts at window 20, a privacy leak at batch 25.

    pip install influxdb-client
    python scripts/seed_dashboard_demo.py             # run_id = demo
    python scripts/seed_dashboard_demo.py --live      # one batch per second
"""
import argparse
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from storage import db  # noqa: E402

COLUMNS = {"adult": (["age", "hours_per_week"], ["education", "occupation"]),
           "credit": (["limit_bal", "bill_amt1"], ["education", "marriage"])}
TRTR = {"adult": 0.90, "credit": 0.77}


def clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def batch(b, rng, run, ts):
    drift, leak = b >= 20, b >= 25
    for ds, (num, cat) in COLUMNS.items():
        shapes = clip(0.88 - (0.12 if drift else 0) + rng.gauss(0, 0.01))
        pairs = clip(0.82 - (0.08 if drift else 0) + rng.gauss(0, 0.015))
        for metric, score in (("column_shapes", shapes), ("column_pair_trends", pairs),
                              ("overall", (shapes + pairs) / 2)):
            db.write("quality", {"run_id": run, "dataset": ds, "metric": metric},
                     {"score": round(score, 4), "batch_id": b}, time=ts)
        for col in num + cat:
            shifted = drift and col in (num[0], cat[0])
            p = rng.uniform(0.0001, 0.01) if shifted else rng.uniform(0.06, 1.0)
            db.write("drift", {"run_id": run, "dataset": ds, "column": col,
                               "test": "ks" if col in num else "chi2"},
                     {"statistic": round(rng.uniform(0.15, 0.3) if shifted else rng.uniform(0.01, 0.06), 4),
                      "p_value": round(p, 5), "alarm": shifted, "window_id": b}, time=ts)
        for model, base in (("logreg", 0.95), ("random_forest", 0.91)):
            ratio = clip(base - (0.07 if drift else 0) + rng.gauss(0, 0.01), 0, 1.2)
            auc = clip(TRTR[ds] * ratio)
            db.write("utility", {"run_id": run, "dataset": ds, "model": model},
                     {"accuracy": round(clip(auc - 0.06), 4), "f1": round(clip(auc - 0.25), 4),
                      "roc_auc": round(auc, 4), "utility_ratio": round(ratio, 4), "batch_id": b}, time=ts)
        db.write("privacy", {"run_id": run, "dataset": ds},
                 {"exact_copy_rate": round(0.05 if leak else rng.uniform(0, 0.002), 4),
                  "median_dcr": round((0.6 if leak else 1.4) + rng.gauss(0, 0.05), 4),
                  "mia_auc": round(clip((0.68 if leak else 0.51) + rng.gauss(0, 0.01)), 4),
                  "batch_id": b}, time=ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="demo")
    ap.add_argument("--batches", type=int, default=35)
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    rng = random.Random(42)
    start = datetime.now(timezone.utc) - timedelta(minutes=args.batches)
    for b in range(1, args.batches + 1):
        ts = datetime.now(timezone.utc) if args.live else start + timedelta(minutes=b)
        batch(b, rng, args.run_id, ts)
        if args.live:
            print(f"batch {b} written"); time.sleep(1)
    print(f"Wrote {args.batches} batches with run_id={args.run_id!r}")


if __name__ == "__main__":
    main()
