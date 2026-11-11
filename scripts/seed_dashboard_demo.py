"""Write fake results to InfluxDB, in every service's real format, so the
dashboard can be shown before the services are connected.
Drift starts at window 20, a privacy leak at window 25.

    pip install influxdb-client
    python scripts/seed_dashboard_demo.py
"""
import argparse
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from storage import db  # noqa: E402

COLUMNS = {"adult": (["age", "hours_per_week"], ["education", "occupation"]),
           "credit-default": (["limit_bal", "bill_amt1"], ["education", "marriage"])}
REAL_BAL_ACC = {"adult": 0.74, "credit-default": 0.66}
DGIM_RATES = {"adult": {"income_gt_50k": 0.24, "age_over_60": 0.08},
              "credit-default": {"default_next_month": 0.22, "education_university": 0.47}}


def clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def window(w, rng, run, ts):
    drift, leak = w >= 20, w >= 25
    for ds, (num, cat) in COLUMNS.items():
        g = lambda base, bump, sd: round(max(0.0, base + (bump if drift else 0) + rng.gauss(0, sd)), 4)
        q = {"n_real": 500, "n_synth": 500, "smd_mean": g(0.08, 0.15, 0.01), "smd_max": g(0.25, 0.45, 0.03),
             "tvd_mean": g(0.03, 0.05, 0.005), "tvd_max": g(0.06, 0.09, 0.008),
             "corr_diff_mean": g(0.04, 0.06, 0.005), "zero_gap_max": g(0.03, 0.02, 0.005),
             "clustream_coverage_ratio": g(1.05, 0.6, 0.05), "denstream_coverage_ratio": g(1.1, 0.5, 0.05)}
        if w % 5 == 0:
            s, p = clip(0.88 - (0.12 if drift else 0) + rng.gauss(0, 0.01)), clip(0.82 - (0.08 if drift else 0) + rng.gauss(0, 0.015))
            q.update(sdmetrics_column_shapes=s, sdmetrics_column_pair_trends=p, sdmetrics_overall=(s + p) / 2)
        db.write(db.QUALITY, {"dataset": ds}, {"window_id": w, **q}, time=ts)
        for col in num + cat:
            kind, metric = ("numeric", "smd") if col in num else ("categorical", "tvd")
            hit = drift and col in (num[0], cat[0])
            db.write(db.COLUMN_QUALITY, {"dataset": ds, "column": col, "kind": kind},
                     {metric: round(rng.uniform(0.3, 0.6) if hit else rng.uniform(0, 0.08), 4)}, time=ts)

        rows = []
        for col in num + cat:
            hit = drift and col in (num[0], cat[0])
            p = rng.uniform(0.0001, 0.005) if hit else rng.uniform(0.06, 1.0)
            rows.append(dict(window_id=w, start=w * 500, end=(w + 1) * 500, dataset=ds, column=col,
                             detector="ks" if col in num else "chi2",
                             statistic=round(rng.uniform(0.15, 0.3) if hit else rng.uniform(0.01, 0.06), 4),
                             p_value=round(p, 5), alarm=int(hit)))
        db._send([db._point(db.DRIFT, {"run_id": run, "dataset": r["dataset"], "column": r["column"],
                                       "detector": r["detector"]},
                            {k: r[k] for k in ("window_id", "start", "end", "statistic", "p_value", "alarm")}, ts)
                  for r in rows])

        real = REAL_BAL_ACC[ds] + rng.gauss(0, 0.005)
        syn = clip(real * (0.93 - (0.08 if drift else 0)) + rng.gauss(0, 0.01))
        db.write(db.UTILITY, {"run_id": run, "dataset": ds}, db._utility_fields(dict(
            window=w, rows_seen=w * 500, acc_synthetic=syn + 0.05, bal_acc_synthetic=syn, f1_synthetic=syn - 0.2,
            acc_real=real + 0.05, bal_acc_real=real, f1_real=real - 0.15, utility_gap=real - syn)), time=ts)
        exact = 0.05 if leak else rng.uniform(0, 0.004)
        db.write(db.PRIVACY, {"run_id": run, "dataset": ds},
                 {"window": w, "rows_seen": w * 500, "exact_copy_rate": exact,
                  "bloom_copy_rate": exact + rng.uniform(0.002, 0.006)}, time=ts)
        for source in ("real", "synthetic"):
            rates = {k: round(clip(v * (0.6 if source == "synthetic" and drift and k == next(iter(DGIM_RATES[ds])) else 1)
                                   + rng.gauss(0, 0.01)), 4) for k, v in DGIM_RATES[ds].items()}
            for pt in [db._point(db.DGIM, {"run_id": run, "dataset": ds, "source": source, "predicate": k},
                                 {"window": 10000, "records_seen": w * 500, "error_bound": 0.5, "rate": v}, ts)
                       for k, v in rates.items()]:
                db._send(pt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="demo")
    ap.add_argument("--windows", type=int, default=35)
    args = ap.parse_args()
    rng = random.Random(42)
    start = datetime.now(timezone.utc) - timedelta(minutes=args.windows)
    for w in range(1, args.windows + 1):
        window(w, rng, args.run_id, start + timedelta(minutes=w))
    print(f"Wrote {args.windows} windows of demo results with run_id={args.run_id!r}")


if __name__ == "__main__":
    main()
