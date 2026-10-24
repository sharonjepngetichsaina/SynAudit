"""Consume a Kafka topic and maintain a uniform reference sample of it.

The quality, privacy and ML-utility services all need to compare incoming
records against real ones, and none of them can hold the whole stream. This
service keeps a bounded uniform sample and writes it to disk periodically, so
the rest of the system has a reference set to query.

Runs in its own consumer group, so it receives a full copy of the stream
independently of the other services.

Examples
--------
    python ingestion/reservoir_service.py --dataset adult --source real --size 1000
    python ingestion/reservoir_service.py --dataset adult --source real --from-start
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

from reservoir import ReservoirSampler

SNAPSHOT_DIR = Path("data/reference")
DATASETS = ("adult", "credit-default")
SOURCES = ("real", "synthetic")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Maintain a reservoir sample of a Kafka topic.")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--source", required=True, choices=SOURCES)
    p.add_argument("--size", type=int, default=1000, help="reservoir size k (default: 1000)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed, for reproducible runs")
    p.add_argument(
        "--snapshot-every",
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


def write_snapshot(sampler: ReservoirSampler, dataset: str, source: str, path: Path) -> None:
    """Persist the current sample. Written to a temporary file and moved into
    place so a reader never sees a half-written snapshot."""
    payload = {
        "dataset": dataset,
        "source": source,
        "reservoir_size": sampler.k,
        "records_seen": sampler.seen,
        "records_held": len(sampler),
        "inclusion_probability": sampler.inclusion_probability,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "sample": sampler.sample(),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    topic = args.topic or f"{args.dataset}-{args.source}"

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path = SNAPSHOT_DIR / f"{args.dataset}-{args.source}-reservoir.json"

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

    sampler = ReservoirSampler(k=args.size, seed=args.seed)
    print(f"consuming {topic} from {args.bootstrap}")
    print(f"reservoir size k={args.size}, seed={args.seed}")

    # Snapshot on Ctrl-C as well as on exit, so a long run is never wasted.
    def on_interrupt(signum, frame):
        write_snapshot(sampler, args.dataset, args.source, snapshot_path)
        print(f"\ninterrupted — snapshot written to {snapshot_path}")
        print(sampler)
        sys.exit(0)

    signal.signal(signal.SIGINT, on_interrupt)

    try:
        for message in consumer:
            record = message.value

            # Store the payload plus its identity: the comparison services need
            # the row, and record_id lets a sampled record be traced back.
            sampler.add(
                {
                    "record_id": record.get("record_id"),
                    "seq": record.get("seq"),
                    **record.get("payload", {}),
                }
            )

            if sampler.seen % args.snapshot_every == 0:
                write_snapshot(sampler, args.dataset, args.source, snapshot_path)
                print(f"  seen {sampler.seen:,}  held {len(sampler):,}  "
                      f"p={sampler.inclusion_probability:.4f}")

    finally:
        consumer.close()

    write_snapshot(sampler, args.dataset, args.source, snapshot_path)
    print(f"\n{sampler}")
    print(f"snapshot: {snapshot_path}")


if __name__ == "__main__":
    main()
