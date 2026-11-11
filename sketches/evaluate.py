"""Accuracy vs memory of the sketches, for the report.

Runs each categorical column of a dataset through Count-Min Sketches of several widths
and HyperLogLogs of several precisions, and compares them with exact counting.
Writes results/sketches/<dataset>-accuracy.csv.

    python -m sketches.evaluate --dataset adult
    python -m sketches.evaluate --dataset adult --source synthetic
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from sketches.count_min import CountMinSketch
from sketches.hyperloglog import HyperLogLog


def evaluate(values, widths, precisions, depth=5):
    exact = Counter(values)
    n, rows = len(values), []
    exact_bytes = sys.getsizeof(exact) + sum(sys.getsizeof(k) for k in exact)
    for w in widths:
        cms = CountMinSketch(w, depth)
        for v in values:
            cms.update(v)
        err = [cms.estimate(k) - c for k, c in exact.items()]
        rows.append(dict(algorithm="count_min", size=w, memory_bytes=cms.memory_bytes, exact_bytes=exact_bytes,
                         max_error_share=max(err) / n, mean_error_share=sum(err) / len(err) / n,
                         bound=cms.epsilon))
    for p in precisions:
        hll = HyperLogLog(p)
        for v in values:
            hll.update(v)
        rows.append(dict(algorithm="hyperloglog", size=1 << p, memory_bytes=hll.memory_bytes, exact_bytes=exact_bytes,
                         estimate=hll.estimate(), exact=len(exact),
                         rel_error=abs(hll.estimate() - len(exact)) / len(exact), bound=hll.standard_error))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("adult", "credit-default"))
    ap.add_argument("--source", default="real", choices=("real", "synthetic"))
    args = ap.parse_args()
    path = f"data/processed/{args.dataset}.csv" if args.source == "real" else f"data/synthetic/{args.dataset}.csv"
    df = pd.read_csv(path)
    out = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        for r in evaluate(df[col].dropna().astype(str).tolist(), widths=(8, 16, 32, 64, 128, 272),
                          precisions=(4, 6, 8, 10, 12)):
            out.append({"dataset": args.dataset, "source": args.source, "column": col, **r})
    res = pd.DataFrame(out)
    Path("results/sketches").mkdir(parents=True, exist_ok=True)
    dest = f"results/sketches/{args.dataset}-{args.source}-accuracy.csv"
    res.to_csv(dest, index=False)
    print(res.groupby(["algorithm", "size"])[["memory_bytes", "max_error_share", "rel_error"]].mean().round(4))
    print(f"saved {dest}")


if __name__ == "__main__":
    main()
