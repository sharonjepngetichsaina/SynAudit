"""Shared InfluxDB helper for SynAudit (owner: Alina).

Every evaluator saves its results through this module, so nobody has to deal
with InfluxDB connections. Example, inside any service:

    from storage import db

    db.write_quality(batch_id=3, dataset="adult", metric="overall", score=0.87)
    db.write_drift(window_id=12, dataset="adult", column="age", test="ks",
                   statistic=0.21, p_value=0.003, alarm=True)
    db.write_utility(batch_id=3, dataset="adult", model="logreg",
                     accuracy=0.81, f1=0.62, roc_auc=0.86, utility_ratio=0.95)
    db.write_privacy(batch_id=3, dataset="adult", exact_copy_rate=0.0,
                     median_dcr=1.4, mia_auc=0.51)

Settings come from environment variables (set in docker-compose):
    INFLUX_URL     default http://localhost:8086 (inside Docker: http://influxdb:8086)
    INFLUX_TOKEN   default synaudit-dev-token
    INFLUX_ORG     default synaudit
    INFLUX_BUCKET  default synaudit
    RUN_ID         experiment label, default "default" (e.g. E1-baseline, E4-leak-5pct)

InfluxDB terms: a *measurement* is like a table; *tags* are labels you filter
by (dataset, model, ...); *fields* are the numbers.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

URL = os.getenv("INFLUX_URL", "http://localhost:8086")
TOKEN = os.getenv("INFLUX_TOKEN", "synaudit-dev-token")
ORG = os.getenv("INFLUX_ORG", "synaudit")
BUCKET = os.getenv("INFLUX_BUCKET", "synaudit")

MEASUREMENTS = ("quality", "drift", "utility", "privacy")

_client = None


def client() -> InfluxDBClient:
    """One shared connection per process, created on first use."""
    global _client
    if _client is None:
        _client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
    return _client


def run_id() -> str:
    return os.getenv("RUN_ID", "default")


def _num(value):
    """numpy numbers -> plain Python numbers (InfluxDB client needs those)."""
    return value.item() if hasattr(value, "item") else value


def write(measurement: str, tags: dict, fields: dict, time: datetime | None = None) -> None:
    """Write one point. Tags are labels, fields are values. None fields are skipped."""
    point = Point(measurement).tag("run_id", tags.pop("run_id", None) or run_id())
    for key, val in tags.items():
        point = point.tag(key, str(val))
    written = 0
    for key, val in fields.items():
        if val is None:
            continue
        val = _num(val)
        point = point.field(key, float(val) if isinstance(val, int) and not isinstance(val, bool) else val)
        written += 1
    if written == 0:
        raise ValueError(f"{measurement}: at least one field must be set")
    point = point.time(time or datetime.now(timezone.utc))
    client().write_api(write_options=SYNCHRONOUS).write(bucket=BUCKET, org=ORG, record=point)


# --- one function per evaluator --------------------------------------------

def write_quality(batch_id: int, dataset: str, metric: str, score: float, run_id: str | None = None):
    """Diana: one point per metric per batch (column_shapes, column_pair_trends, overall)."""
    write("quality", {"run_id": run_id, "dataset": dataset, "metric": metric},
          {"score": score, "batch_id": batch_id})


def write_drift(window_id: int, dataset: str, column: str, test: str, statistic: float | None = None,
                p_value: float | None = None, alarm: bool = False, run_id: str | None = None):
    """Grace: one point per column per window. test = ks, chi2, adwin, page_hinkley."""
    write("drift", {"run_id": run_id, "dataset": dataset, "column": column, "test": test},
          {"statistic": statistic, "p_value": p_value, "alarm": bool(alarm), "window_id": window_id})


def write_utility(batch_id: int, dataset: str, model: str, accuracy: float | None = None,
                  f1: float | None = None, roc_auc: float | None = None,
                  utility_ratio: float | None = None, run_id: str | None = None):
    """Sharon: one point per model per batch. utility_ratio = TSTR AUC / TRTR AUC."""
    write("utility", {"run_id": run_id, "dataset": dataset, "model": model},
          {"accuracy": accuracy, "f1": f1, "roc_auc": roc_auc,
           "utility_ratio": utility_ratio, "batch_id": batch_id})


def write_privacy(batch_id: int, dataset: str, exact_copy_rate: float | None = None,
                  median_dcr: float | None = None, dcr_train_share: float | None = None,
                  mia_auc: float | None = None, run_id: str | None = None):
    """Sharon: one point per batch."""
    write("privacy", {"run_id": run_id, "dataset": dataset},
          {"exact_copy_rate": exact_copy_rate, "median_dcr": median_dcr,
           "dcr_train_share": dcr_train_share, "mia_auc": mia_auc, "batch_id": batch_id})


# --- reading (dashboard) ----------------------------------------------------

def read(measurement: str, dataset: str | None = None, run: str | None = None, since: str = "-30d"):
    """Return one measurement as a pandas DataFrame, one row per point, oldest first."""
    import pandas as pd

    if measurement not in MEASUREMENTS:
        raise ValueError(f"unknown measurement {measurement!r}")
    flt = [f'r._measurement == "{measurement}"']
    if dataset:
        flt.append(f'r.dataset == "{dataset}"')
    if run:
        flt.append(f'r.run_id == "{run}"')
    flux = (f'from(bucket: "{BUCKET}") |> range(start: {since}) '
            f'|> filter(fn: (r) => {" and ".join(flt)}) '
            f'|> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value") '
            f'|> group()')
    df = client().query_api().query_data_frame(flux, org=ORG)
    if isinstance(df, list):
        df = pd.concat(df, ignore_index=True) if df else pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    df = df.drop(columns=[c for c in ("result", "table", "_start", "_stop", "_measurement") if c in df.columns])
    df = df.rename(columns={"_time": "time"}).sort_values("time").reset_index(drop=True)
    for col in ("batch_id", "window_id"):
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    return df


def tag_values(tag: str) -> list[str]:
    """All values of a tag (dataset or run_id) across SynAudit, for dashboard filters."""
    flux = (f'import "influxdata/influxdb/schema" '
            f'schema.tagValues(bucket: "{BUCKET}", tag: "{tag}", start: -30d)')
    tables = client().query_api().query(flux, org=ORG)
    return sorted({rec.get_value() for table in tables for rec in table.records})
