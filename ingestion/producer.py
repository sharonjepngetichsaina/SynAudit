"""Replay a canonical CSV into its Kafka topic as a stream.

Implements the envelope defined in docs/stream-contract.md. Real and synthetic
streams run as separate processes at independent rates; they are not synchronised.

Examples
--------
    # stream the real Adult data at 50 records/second
    python ingestion/producer.py --dataset adult --source real --rate 50

    # check the record format without a broker
    python ingestion/producer.py --dataset adult --source real --limit 3 --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA_VERSION = "1.0"
PROCESSED = Path("data/processed")

DATASETS = ("adult", "credit-default")
SOURCES = ("real", "synthetic")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay a canonical CSV into Kafka.")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--source", required=True, choices=SOURCES)
    p.add_argument(
        "--rate",
        type=float,
        default=50.0,
        help="records per second; 0 means as fast as possible (default: 50)",
    )
    p.add_argument("--limit", type=int, default=None, help="stop after N records")
    p.add_argument(
        "--bootstrap",
        default="localhost:29092",
        help="broker address: localhost:29092 from the host, kafka:9092 from a container",
    )
    p.add_argument("--topic", default=None, help="override the derived topic name")
    p.add_argument("--dry-run", action="store_true", help="print records instead of sending")
    return p.parse_args()


def source_path(dataset: str, source: str) -> Path:
    """Real data comes from data/processed/, synthetic from data/synthetic/."""
    if source == "real":
        return PROCESSED / f"{dataset}.csv"
    return Path("data/synthetic") / f"{dataset}.csv"


def clean_value(value):
    """Make a pandas value JSON-safe.

    pandas uses NaN/NaT for missing data and numpy scalar types for numbers;
    neither survives json.dumps. The contract requires null for missing.
    """
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def build_record(dataset: str, source: str, seq: int, row: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "source": source,
        "record_id": f"{dataset}-{source}-{seq:06d}",
        "seq": seq,
        "emitted_at": datetime.now(timezone.utc).isoformat(),
        "payload": {k: clean_value(v) for k, v in row.items()},
    }


def main() -> None:
    args = parse_args()

    path = source_path(args.dataset, args.source)
    if not path.exists():
        sys.exit(
            f"{path} not found.\n"
            "Real data: run python ingestion/preprocess.py first.\n"
            "Synthetic data: waiting on the generation step."
        )

    topic = args.topic or f"{args.dataset}-{args.source}"
    df = pd.read_csv(path)
    if args.limit:
        df = df.head(args.limit)

    interval = 1.0 / args.rate if args.rate > 0 else 0.0

    producer = None
    if not args.dry_run:
        from kafka import KafkaProducer

        producer = KafkaProducer(
            bootstrap_servers=args.bootstrap,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8"),
            acks="all",          # wait for durable commit rather than fire-and-forget
            linger_ms=5,         # small batching window, still near-real-time
            retries=3,
        )
        print(f"connected to {args.bootstrap}")

    print(f"streaming {len(df):,} records -> {topic} at {args.rate or 'max'} rec/s")
    started = time.monotonic()
    sent = 0

    try:
        for seq, row in enumerate(df.to_dict(orient="records")):
            record = build_record(args.dataset, args.source, seq, row)

            if args.dry_run:
                print(json.dumps(record, indent=2))
            else:
                producer.send(topic, key=record["record_id"], value=record)

            sent += 1
            if sent % 1000 == 0:
                print(f"  {sent:,} sent")
            if interval:
                time.sleep(interval)

    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if producer is not None:
            # Sends are buffered; without this the tail of the stream is lost.
            producer.flush()
            producer.close()

    elapsed = time.monotonic() - started
    throughput = sent / elapsed if elapsed else 0.0
    print(f"done: {sent:,} records in {elapsed:.1f}s ({throughput:.0f} rec/s)")


if __name__ == "__main__":
    main()
