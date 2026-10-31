"""Sketch service (owner: Alina).

Reads the real and the synthetic stream of one dataset from Kafka, keeps a
Count-Min Sketch and a HyperLogLog per categorical column for each stream, and every
--report-every records writes to InfluxDB:

  sketches     one point per (source, column): HLL distinct estimate, CMS error,
               exact values and memory, so the sketches' accuracy can be evaluated
  sketch_freq  one point per (column, category): estimated share in the real and in
               the synthetic stream, so the dashboard shows where the generator differs

Topics and record format: docs/stream-contract.md. Consumer group: synaudit-sketches.

    python -m sketches.sketch_service --dataset adult --bootstrap localhost:29092
    python -m sketches.sketch_service --dataset adult --from-csv          # no Kafka needed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sketches.tracker import StreamSketches  # noqa: E402
from storage import db  # noqa: E402

SOURCES = ("real", "synthetic")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Count-Min Sketch and HyperLogLog over the SynAudit streams.")
    p.add_argument("--dataset", required=True, choices=("adult", "credit-default"))
    p.add_argument("--bootstrap", default="localhost:29092", help="kafka:9092 inside Docker")
    p.add_argument("--report-every", type=int, default=1000, help="records between database writes")
    p.add_argument("--cms-width", type=int, default=272, help="w = e / epsilon (272 -> epsilon 0.01)")
    p.add_argument("--cms-depth", type=int, default=5, help="d = ln(1 / delta) (5 -> delta 0.007)")
    p.add_argument("--hll-precision", type=int, default=10, help="m = 2^p registers (10 -> 1.04/32 = 3.3%%)")
    p.add_argument("--from-csv", action="store_true", help="replay data/processed and data/synthetic CSVs")
    p.add_argument("--idle-timeout", type=float, default=None, help="stop after this many idle seconds")
    p.add_argument("--no-db", action="store_true", help="print summaries instead of writing to InfluxDB")
    return p.parse_args(argv)


def records_from_csv(dataset: str):
    """Interleave the two CSV files the producer would replay, one record from each in turn."""
    import pandas as pd

    frames = {"real": pd.read_csv(f"data/processed/{dataset}.csv"),
              "synthetic": pd.read_csv(f"data/synthetic/{dataset}.csv")}
    iters = {s: df.where(df.notna(), None).to_dict("records") for s, df in frames.items()}
    for i in range(max(len(v) for v in iters.values())):
        for source in SOURCES:
            if i < len(iters[source]):
                yield {"dataset": dataset, "source": source, "seq": i, "payload": iters[source][i]}


def records_from_kafka(args):
    from kafka import KafkaConsumer

    consumer = KafkaConsumer(
        *(f"{args.dataset}-{s}" for s in SOURCES),
        bootstrap_servers=args.bootstrap, group_id="synaudit-sketches",
        auto_offset_reset="earliest",                       # stream contract, section 3
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        consumer_timeout_ms=int(args.idle_timeout * 1000) if args.idle_timeout else float("inf"))
    for msg in consumer:
        yield msg.value


def report(dataset, streams, run_id=None, write=True):
    points = []
    columns = sorted(set(streams["real"].columns) | set(streams["synthetic"].columns))
    for source, s in streams.items():
        for col in s.columns:
            sm = s.summary(col)
            points.append(db._point(db.SKETCHES, {"run_id": run_id, "dataset": dataset, "source": source,
                                                  "column": col}, sm))
    for col in columns:
        cats = set()
        for s in streams.values():
            if col in s.columns and s.columns[col].exact is not None:
                cats |= set(s.columns[col].exact)
        real, syn = streams["real"].shares(col, cats), streams["synthetic"].shares(col, cats)
        for cat in cats:
            exact_real = streams["real"].columns[col].exact.get(cat, 0) / max(streams["real"].n, 1) \
                if col in streams["real"].columns else None
            points.append(db._point(db.SKETCH_FREQ, {"run_id": run_id, "dataset": dataset, "column": col,
                                                     "category": cat},
                                    {"real_share": real.get(cat), "synthetic_share": syn.get(cat),
                                     "real_exact_share": exact_real,
                                     "records_seen": streams["real"].n + streams["synthetic"].n}))
    if write and points:
        db._send(points)
    return len(points)


def main(argv=None):
    args = parse_args(argv)
    streams = {s: StreamSketches(args.cms_width, args.cms_depth, args.hll_precision) for s in SOURCES}
    source = records_from_csv(args.dataset) if args.from_csv else records_from_kafka(args)
    seen = 0
    for rec in source:
        if rec.get("dataset") != args.dataset or rec.get("source") not in streams:
            continue
        streams[rec["source"]].update(rec["payload"])
        seen += 1
        if seen % args.report_every == 0:
            n = report(args.dataset, streams, write=not args.no_db)
            print(f"{seen:,} records  real={streams['real'].n:,} synthetic={streams['synthetic'].n:,}  -> {n} points")
    if seen % args.report_every:
        report(args.dataset, streams, write=not args.no_db)
    print(f"done: {seen:,} records")


if __name__ == "__main__":
    main()
