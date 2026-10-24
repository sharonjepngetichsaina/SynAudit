"""Consume a Kafka topic and track sliding-window rates with DGIM.

Monitors binary properties of the recent stream — the share of high-income
records, the share of defaults, the share of records with a missing value —
in memory that stays bounded however long the stream runs.

These rates are the cheap early signal: a shift in the positive-class rate or a
jump in missingness shows up here before any distributional test runs.

Runs in its own consumer group, so it sees a full copy of the stream.

Examples
--------
    python ingestion/dgim_service.py --dataset adult --source real --window 1000
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from dgim import DGIMCounter

SNAPSHOT_DIR = Path("data/reference")
DATASETS = ("adult", "credit-default")
SOURCES = ("real", "synthetic")

# Binary properties worth watching per dataset. Each takes a payload dict and
# returns True or False. Adding one here is all it takes to monitor it.
PREDICATES: dict[str, dict[str, Callable[[dict], bool]]] = {
    "adult": {
        "high_income": lambda r: r.get("income") == ">50K",
        "missing_any": lambda r: any(v is None for v in r.values()),
        "missing_occupation": lambda r: r.get("occupation") is None,
        "full_time": lambda r: (r.get("hours_per_week") or 0) >= 40,
    },
    "credit-default": {
        "default": lambda r: r.get("default_payment_next_month") == 1,
        "missing_any": lambda r: any(v is None for v in r.values()),
        "high_limit": lambda r: (r.get("limit_bal") or 0) >= 200000,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track sliding-window rates with DGIM.")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--source", required=True, choices=SOURCES)
    p.add_argument("--window", type=int, default=1000, help="window size N (default: 1000)")
    p.add_argument(
        "--buckets-per-size",
        type=int,
        default=2,
        help="r: error bound is 1/(r-1), so larger is tighter but uses more memory (default: 2)",
    )
    p.add_argument(
        "--report-every",
        type=int,
        default=500,
        help="write a snapshot every N records (default: 500)",
    )
    p.add_argument(
        "--bootstrap",
        default="localhost:29092",
        help="broker address: localhost:29092 from the host, kafka:9092 from a container",
    )
    p.add_argument("--topic", default=None, help="override the derived topic name")
    p.add_argument(
        "--from-start",
        action="store_true",
        help="read the topic from the beginning, ignoring any committed offset",
    )
    p.add_argument(
        "--idle-timeout",
        type=int,
        default=30,
        help="seconds with no new records before exiting; 0 means run forever",
    )
    return p.parse_args()


def rates(counters: dict[str, DGIMCounter], window: int, seen: int) -> dict[str, dict]:
    """Current estimated rate for each tracked property.

    The denominator is the window size, or the number of records seen while the
    window is still filling — otherwise early rates read far too low.
    """
    divisor = min(window, seen) or 1
    return {
        name: {
            "count_estimate": c.count(),
            "rate": c.count() / divisor,
            "buckets": c.memory_buckets,
        }
        for name, c in counters.items()
    }


def write_snapshot(
    counters: dict[str, DGIMCounter],
    dataset: str,
    source: str,
    window: int,
    seen: int,
    error_bound: float,
    path: Path,
) -> None:
    payload = {
        "dataset": dataset,
        "source": source,
        "window": window,
        "records_seen": seen,
        "error_bound": error_bound,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "rates": rates(counters, window, seen),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    topic = args.topic or f"{args.dataset}-{args.source}"

    predicates = PREDICATES[args.dataset]
    counters = {
        name: DGIMCounter(window=args.window, buckets_per_size=args.buckets_per_size)
        for name in predicates
    }

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path = SNAPSHOT_DIR / f"{args.dataset}-{args.source}-dgim.json"

    from kafka import KafkaConsumer

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=args.bootstrap,
        group_id=None if args.from_start else "synaudit-ingest-stats",
        auto_offset_reset="earliest",
        enable_auto_commit=not args.from_start,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        consumer_timeout_ms=args.idle_timeout * 1000 if args.idle_timeout else -1,
    )

    seen = 0
    error_bound = next(iter(counters.values())).error_bound

    print(f"consuming {topic} from {args.bootstrap}")
    print(f"window N={args.window:,}, r={args.buckets_per_size}, "
          f"error bound {error_bound:.0%}")
    print(f"tracking: {', '.join(predicates)}")

    def report() -> None:
        current = rates(counters, args.window, seen)
        parts = [f"{name} {v['rate']:.3f}" for name, v in current.items()]
        total_buckets = sum(v["buckets"] for v in current.values())
        print(f"  seen {seen:,}  " + "  ".join(parts) + f"   [{total_buckets} buckets]")

    def on_interrupt(signum, frame):
        write_snapshot(counters, args.dataset, args.source, args.window,
                       seen, error_bound, snapshot_path)
        print(f"\ninterrupted — snapshot written to {snapshot_path}")
        sys.exit(0)

    signal.signal(signal.SIGINT, on_interrupt)

    try:
        for message in consumer:
            payload = message.value.get("payload", {})
            seen += 1

            for name, predicate in predicates.items():
                counters[name].add(1 if predicate(payload) else 0)

            if seen % args.report_every == 0:
                report()
                write_snapshot(counters, args.dataset, args.source, args.window,
                               seen, error_bound, snapshot_path)
    finally:
        consumer.close()

    report()
    write_snapshot(counters, args.dataset, args.source, args.window,
                   seen, error_bound, snapshot_path)
    print(f"\nsnapshot: {snapshot_path}")


if __name__ == "__main__":
    main()
