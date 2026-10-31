"""Shared InfluxDB helper for SynAudit (owner: Alina).

Each service saves its results with one call; the dashboard reads them back.
The schemas follow what each service already produces, so nobody has to change
their own result format:

    Diana   quality      quality/sinks.py writes synaudit_quality itself (docs/INFLUX_SCHEMA.md)
    Grace   drift        db.write_drift_rows(window_tests(...))         -> drift_events
    Sharon  utility      db.write_utility_rows(df)  (columns of adult-utility.csv)  -> utility
    Sharon  privacy      db.write_privacy_rows(df)  (columns of adult-copies.csv)   -> privacy
    Matilda DGIM         db.write_dgim_snapshot(payload)  (the JSON dgim_service writes) -> dgim

Settings (environment variables, defaults work on a laptop):
    INFLUX_URL     http://localhost:8086   (inside Docker: http://influxdb:8086)
    INFLUX_TOKEN   synaudit-dev-token
    INFLUX_ORG     synaudit
    INFLUX_BUCKET  synaudit
    RUN_ID         experiment label, default "default" (e.g. E1-baseline)

Datasets are named like the Kafka topics: "adult" and "credit-default".
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

URL = os.getenv("INFLUX_URL", "http://localhost:8086")
TOKEN = os.getenv("INFLUX_TOKEN", "synaudit-dev-token")
ORG = os.getenv("INFLUX_ORG", "synaudit")
BUCKET = os.getenv("INFLUX_BUCKET", "synaudit")

QUALITY = "synaudit_quality"                # Diana, written by quality/sinks.py
COLUMN_QUALITY = "synaudit_column_quality"  # Diana, written by quality/sinks.py
DRIFT = "drift_events"                      # Grace
UTILITY = "utility"                         # Sharon
PRIVACY = "privacy"                         # Sharon
DGIM = "dgim"                               # Matilda
SKETCHES = "sketches"                       # Alina: HLL / CMS accuracy per column
SKETCH_FREQ = "sketch_freq"                 # Alina: category shares, real vs synthetic
MEASUREMENTS = (QUALITY, COLUMN_QUALITY, DRIFT, UTILITY, PRIVACY, DGIM, SKETCHES, SKETCH_FREQ)

# Counters are stored as integers. InfluxDB refuses a field whose type changes,
# so these are always written as int, everything else as float.
INT_FIELDS = {"window_id", "window", "n_real", "n_synth", "start", "end",
              "rows_seen", "records_seen", "alarm", "hll_exact",
              "hll_memory_bytes", "cms_memory_bytes", "exact_memory_bytes"}

_client = None


def client() -> InfluxDBClient:
    global _client
    if _client is None:
        _client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
    return _client


def run_id() -> str:
    return os.getenv("RUN_ID", "default")


def _plain(v):
    return v.item() if hasattr(v, "item") else v


def _point(measurement: str, tags: dict, fields: dict, time=None):
    p = Point(measurement)
    if measurement not in (QUALITY, COLUMN_QUALITY):   # Diana's schema has no run_id
        p = p.tag("run_id", tags.pop("run_id", None) or run_id())
    tags.pop("run_id", None)
    for k, v in tags.items():
        if v is not None:
            p = p.tag(k, str(v))
    n = 0
    for k, v in fields.items():
        v = _plain(v)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if isinstance(v, bool):
            v = int(v) if k in INT_FIELDS else v
        elif isinstance(v, (int, float)):
            v = int(round(v)) if k in INT_FIELDS else float(v)
        p = p.field(k, v)
        n += 1
    if n == 0:
        raise ValueError(f"{measurement}: no field to write")
    return p.time(time or datetime.now(timezone.utc))


def _send(points) -> None:
    client().write_api(write_options=SYNCHRONOUS).write(bucket=BUCKET, org=ORG, record=points)


def write(measurement: str, tags: dict, fields: dict, time=None) -> None:
    """Low-level: one point. Tags are labels to filter by, fields are values."""
    _send(_point(measurement, dict(tags), fields, time))


# --- Diana: quality (her service writes it itself; this is for other code) ---

def write_quality(window_id: int, dataset: str, **scores):
    write(QUALITY, {"dataset": dataset}, {"window_id": window_id, **scores})


# --- Grace: drift (rows shaped like drift/window_tests.py returns) ----------

def write_drift(window_id, dataset, column, detector, statistic=None, p_value=None,
                alarm=0, start=None, end=None, run_id=None):
    write(DRIFT, {"run_id": run_id, "dataset": dataset, "column": column, "detector": detector},
          {"window_id": window_id, "start": start, "end": end,
           "statistic": statistic, "p_value": p_value, "alarm": alarm})


def write_drift_rows(df, run_id=None) -> int:
    """All rows of window_tests(...) at once. Returns rows written."""
    pts = [_point(DRIFT, {"run_id": run_id, "dataset": r["dataset"], "column": r["column"],
                          "detector": r["detector"]},
                  {k: r.get(k) for k in ("window_id", "start", "end", "statistic", "p_value", "alarm")})
           for r in df.to_dict("records")]
    if pts:
        _send(pts)
    return len(pts)


# --- Sharon: utility and privacy (columns of her CSV files) ------------------

UTILITY_FIELDS = ("window", "rows_seen", "acc_synthetic", "bal_acc_synthetic", "f1_synthetic",
                  "acc_real", "bal_acc_real", "f1_real", "utility_gap")
PRIVACY_FIELDS = ("window", "rows_seen", "bloom_copy_rate", "exact_copy_rate")


def _utility_fields(r):
    f = {k: r.get(k) for k in UTILITY_FIELDS}
    syn, real = _plain(f.get("bal_acc_synthetic")), _plain(f.get("bal_acc_real"))
    if syn is not None and real:      # utility ratio = balanced accuracy synthetic / real
        f["utility_ratio"] = syn / real
    return f


def write_utility(window, dataset, run_id=None, **values):
    write(UTILITY, {"run_id": run_id, "dataset": dataset}, _utility_fields({"window": window, **values}))


def write_utility_rows(df, run_id=None) -> int:
    pts = [_point(UTILITY, {"run_id": run_id, "dataset": r["dataset"]}, _utility_fields(r))
           for r in df.to_dict("records")]
    if pts:
        _send(pts)
    return len(pts)


def write_privacy(window, dataset, run_id=None, **values):
    write(PRIVACY, {"run_id": run_id, "dataset": dataset},
          {k: v for k, v in {"window": window, **values}.items() if k in PRIVACY_FIELDS})


def write_privacy_rows(df, run_id=None) -> int:
    pts = [_point(PRIVACY, {"run_id": run_id, "dataset": r["dataset"]},
                  {k: r.get(k) for k in PRIVACY_FIELDS}) for r in df.to_dict("records")]
    if pts:
        _send(pts)
    return len(pts)


# --- Matilda: DGIM snapshot (the payload dgim_service.write_snapshot builds) --

def _rate_values(value):
    """A rate entry may be a number or a dict of numbers; keep its numbers."""
    if isinstance(value, dict):
        nums = {k: _plain(v) for k, v in value.items()
                if isinstance(_plain(v), (int, float)) and not isinstance(_plain(v), bool)}
        if "rate" not in nums and len(nums) == 1:
            nums = {"rate": next(iter(nums.values()))}
        return {k: float(v) for k, v in nums.items()}
    return {"rate": float(_plain(value))}


def write_dgim_snapshot(payload: dict, run_id=None) -> int:
    """One point per tracked predicate: dataset, source (real/synthetic), predicate, rate."""
    meta = {"window": payload.get("window"), "records_seen": payload.get("records_seen"),
            "error_bound": payload.get("error_bound")}
    pts = [_point(DGIM, {"run_id": run_id, "dataset": payload["dataset"],
                         "source": payload["source"], "predicate": name},
                  {**meta, **_rate_values(value)})
           for name, value in (payload.get("rates") or {}).items()]
    if pts:
        _send(pts)
    return len(pts)


# --- reading (dashboard) -----------------------------------------------------

def read(measurement: str, dataset: str | None = None, run: str | None = None, since: str = "-30d"):
    import pandas as pd

    if measurement not in MEASUREMENTS:
        raise ValueError(f"unknown measurement {measurement!r}")
    flt = [f'r._measurement == "{measurement}"']
    if dataset:
        flt.append(f'r.dataset == "{dataset}"')
    if run:
        flt.append(f'(not exists r.run_id or r.run_id == "{run}")')
    flux = (f'from(bucket: "{BUCKET}") |> range(start: {since}) '
            f'|> filter(fn: (r) => {" and ".join(flt)}) '
            f'|> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value") '
            f'|> group()')
    df = client().query_api().query_data_frame(flux, org=ORG)
    if isinstance(df, list):
        df = pd.concat(df, ignore_index=True) if df else pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    df = df.drop(columns=[c for c in ("result", "table", "_start", "_stop", "_measurement") if c in df])
    df = df.rename(columns={"_time": "time"}).sort_values("time").reset_index(drop=True)
    for col in INT_FIELDS & set(df.columns):
        df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")
    return df


def tag_values(tag: str) -> list[str]:
    flux = (f'import "influxdata/influxdb/schema" '
            f'schema.tagValues(bucket: "{BUCKET}", tag: "{tag}", start: -30d)')
    tables = client().query_api().query(flux, org=ORG)
    return sorted({rec.get_value() for t in tables for rec in t.records})
