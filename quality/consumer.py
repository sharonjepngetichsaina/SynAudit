"""Streaming quality consumer.

Reads <dataset>-real and <dataset>-synthetic from Kafka, keeps a sliding window of
the latest records of each, and every `--step` synthetic records scores the windows
(distribution, correlation, consistency, SDMetrics, CluStream/DenStream).

Live, from the host (Kafka is published on localhost:29092):
    python quality/consumer.py --dataset adult --bootstrap localhost:29092

Inside Docker (bootstrap defaults to the KAFKA_BOOTSTRAP env var, kafka:9092):
    python -m quality.consumer --dataset adult

Offline replay without Kafka (same monitor code, useful for tests and demos):
    python quality/consumer.py --dataset adult --from-csv --max-windows 20
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from quality import metrics as m  # noqa: E402
from quality.monitor import QualityMonitor, format_line  # noqa: E402
from quality.sinks import CsvSink, InfluxSink  # noqa: E402

log = logging.getLogger("synaudit.consumer")

WRAPPER_KEYS = ("record", "data", "row", "values", "features", "payload")


def extract_record(payload, columns):
    """Turn one Kafka message value into {column: value}.

    Accepts a flat JSON object, or an object that wraps the row under one of
    record/data/row/values/features/payload. Returns None if the row is unusable.
    """
    if isinstance(payload, dict):
        for key in WRAPPER_KEYS:
            if isinstance(payload.get(key), dict):
                payload = payload[key]
                break
    if isinstance(payload, list) and len(payload) == len(columns):
        payload = dict(zip(columns, payload))
    if not isinstance(payload, dict):
        return None
    found = [c for c in columns if c in payload]
    if len(found) < max(1, len(columns) // 2):
        return None
    return {c: payload.get(c) for c in columns}


def build_arg_parser():
    env = os.environ.get
    p = argparse.ArgumentParser(description="SynAudit streaming quality monitor")
    p.add_argument("--dataset", choices=["adult", "credit-default"], default=env("DATASET"), required=not env("DATASET"))
    p.add_argument("--bootstrap", default=env("KAFKA_BOOTSTRAP", "localhost:29092"),
                   help="localhost:29092 from the host, kafka:9092 from a container")
    p.add_argument("--topic-real", default=None, help="default <dataset>-real")
    p.add_argument("--topic-synthetic", default=None, help="default <dataset>-synthetic")
    p.add_argument("--group-id", default=None)
    p.add_argument("--offset", choices=["earliest", "latest"], default=env("QUALITY_OFFSET", "earliest"))
    p.add_argument("--window", type=int, default=int(env("QUALITY_WINDOW", 500)), help="records kept per source")
    p.add_argument("--step", type=int, default=int(env("QUALITY_STEP", 100)), help="new synthetic records between windows")
    p.add_argument("--sdmetrics-every", type=int, default=int(env("QUALITY_SDMETRICS_EVERY", 5)),
                   help="run the (slower) SDMetrics report every N windows; 0 = never")
    p.add_argument("--no-clustering", action="store_true")
    p.add_argument("--k-macro", type=int, default=5, help="k for the CluStream macro-clustering")
    p.add_argument("--metadata", default=None, help="SDV metadata JSON (default data/synthetic/<dataset>-metadata.json)")
    p.add_argument("--out", default=None, help="CSV output (default results/stream/<dataset>-windows.csv)")
    p.add_argument("--max-windows", type=int, default=0, help="stop after N windows (0 = run forever)")
    p.add_argument("--idle-timeout", type=float, default=float(env("QUALITY_IDLE_TIMEOUT", 0)),
                   help="stop when no message arrived for N seconds (0 = never)")
    p.add_argument("--from-csv", action="store_true", help="replay data/processed + data/synthetic CSVs, no Kafka")
    p.add_argument("--no-influx", action="store_true", help="ignore INFLUX_* environment variables")
    return p


def load_types(args, real_df=None):
    meta = Path(args.metadata) if args.metadata else ROOT / "data" / "synthetic" / f"{args.dataset}-metadata.json"
    if meta.exists():
        return m.load_column_types(meta)
    if real_df is not None:
        log.warning("metadata %s not found; inferring column types from the data", meta)
        return m.infer_column_types(real_df)
    raise SystemExit(f"Metadata file not found: {meta}. Run generate_synthetic.py or pass --metadata.")


def make_sinks(args):
    out = Path(args.out) if args.out else ROOT / "results" / "stream" / f"{args.dataset}-windows.csv"
    sinks = [CsvSink(out)]
    log.info("writing window scores to %s", out)
    if not args.no_influx:
        try:
            influx = InfluxSink.from_env()
        except ImportError:
            influx = None
            log.warning("INFLUX_URL is set but influxdb-client is not installed; InfluxDB output disabled")
        if influx is not None:
            sinks.append(influx)
            log.info("writing window scores to InfluxDB %s", os.environ.get("INFLUX_URL"))
    return sinks


def emit(result, sinks):
    print(format_line(result), flush=True)
    for sink in sinks:
        try:
            sink.write(result)
        except Exception as exc:  # a sink problem must not stop the monitor
            log.error("%s failed: %s", type(sink).__name__, exc)


def run_csv(args):
    real_path = ROOT / "data" / "processed" / f"{args.dataset}.csv"
    synth_path = ROOT / "data" / "synthetic" / f"{args.dataset}-synthetic.csv"
    real = pd.read_csv(real_path)
    synth = pd.read_csv(synth_path)
    ctypes = load_types(args, real)
    monitor = make_monitor(args, ctypes)
    sinks = make_sinks(args)
    real_rows = real.astype(object).where(real.notna(), None).to_dict("records")
    synth_rows = synth.astype(object).where(synth.notna(), None).to_dict("records")
    count = 0
    for i in range(max(len(real_rows), len(synth_rows))):
        for source, rows in (("real", real_rows), ("synthetic", synth_rows)):
            if i < len(rows):
                res = monitor.add(source, rows[i])
                if res:
                    emit(res, sinks)
                    count += 1
                    if args.max_windows and count >= args.max_windows:
                        return
    log.info("replay finished after %d windows", count)


def make_monitor(args, ctypes):
    return QualityMonitor(args.dataset, ctypes, window=args.window, step=args.step,
                          sdmetrics_every=args.sdmetrics_every, clustering=not args.no_clustering,
                          k_macro=args.k_macro)


def run_kafka(args):
    try:
        from kafka import KafkaConsumer
    except ImportError:
        raise SystemExit("kafka-python is not installed: pip install kafka-python")

    ctypes = load_types(args)
    monitor = make_monitor(args, ctypes)
    sinks = make_sinks(args)
    topics = {
        (args.topic_real or f"{args.dataset}-real"): "real",
        (args.topic_synthetic or f"{args.dataset}-synthetic"): "synthetic",
    }
    consumer = KafkaConsumer(
        *topics.keys(),
        bootstrap_servers=args.bootstrap,
        group_id=args.group_id or f"synaudit-quality-{args.dataset}",
        auto_offset_reset=args.offset,
        enable_auto_commit=True,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        consumer_timeout_ms=1000,
    )
    log.info("consuming %s from %s (window=%d, step=%d)", list(topics), args.bootstrap, args.window, args.step)

    counts, bad, windows = {"real": 0, "synthetic": 0}, 0, 0
    last_msg = time.time()
    last_status = 0.0
    try:
        while True:
            batches = consumer.poll(timeout_ms=500, max_records=500)
            for tp, messages in batches.items():
                source = topics[tp.topic]
                for msg in messages:
                    last_msg = time.time()
                    record = extract_record(msg.value, ctypes.columns)
                    if record is None:
                        bad += 1
                        if bad == 1:
                            log.warning("message does not match the dataset columns, skipping: %.200s", msg.value)
                        continue
                    counts[source] += 1
                    res = monitor.add(source, record)
                    if res:
                        emit(res, sinks)
                        windows += 1
                        if args.max_windows and windows >= args.max_windows:
                            return
            now = time.time()
            if not batches and now - last_status > 15 and windows == 0:
                log.info("waiting for data: real=%d synthetic=%d (need %d of each)",
                         counts["real"], counts["synthetic"], monitor.min_fill)
                last_status = now
            if args.idle_timeout and now - last_msg > args.idle_timeout:
                log.info("no messages for %.0fs, stopping", args.idle_timeout)
                return
    except KeyboardInterrupt:
        log.info("stopped by user")
    finally:
        consumer.close()
        for s in sinks:
            s.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args()
    if args.from_csv:
        run_csv(args)
    else:
        run_kafka(args)


if __name__ == "__main__":
    main()
