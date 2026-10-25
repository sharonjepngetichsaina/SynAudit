"""Where window results go: a CSV file (always) and InfluxDB (when configured)."""
from __future__ import annotations

import csv
import logging
import math
import os
from pathlib import Path

log = logging.getLogger("synaudit.sinks")

MEASUREMENT_WINDOW = "synaudit_quality"
MEASUREMENT_COLUMN = "synaudit_column_quality"


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


class CsvSink:
    """One row per window. Column set is fixed by the first row (all score keys are always present)."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fields = None
        self._new = not self.path.exists() or self.path.stat().st_size == 0

    def write(self, result):
        row = result.flat()
        if self._fields is None:
            self._fields = list(row.keys())
            if not self._new:   # appending to an old file: keep its header
                with open(self.path, newline="", encoding="utf-8") as fh:
                    self._fields = next(csv.reader(fh))
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self._fields, extrasaction="ignore")
            if self._new:
                w.writeheader()
                self._new = False
            w.writerow({k: ("" if (isinstance(v, float) and math.isnan(v)) else v) for k, v in row.items()})

    def close(self):
        pass


class InfluxSink:
    """Writes two measurements (schema documented in docs/INFLUX_SCHEMA.md).

    synaudit_quality         tags: dataset                      fields: window_id + every score
    synaudit_column_quality  tags: dataset, column, kind        fields: smd | tvd | zero_gap
    """

    def __init__(self, url, token, org, bucket):
        from influxdb_client import InfluxDBClient
        from influxdb_client.client.write_api import SYNCHRONOUS
        self.bucket, self.org = bucket, org
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)

    @classmethod
    def from_env(cls):
        url = os.environ.get("INFLUX_URL")
        if not url:
            return None
        return cls(url, os.environ.get("INFLUX_TOKEN", ""), os.environ.get("INFLUX_ORG", "synaudit"),
                   os.environ.get("INFLUX_BUCKET", "synaudit"))

    def write(self, result):
        from influxdb_client import Point
        points = []
        p = Point(MEASUREMENT_WINDOW).tag("dataset", result.dataset).field("window_id", int(result.window_id))
        p = p.field("n_real", int(result.n_real)).field("n_synth", int(result.n_synth))
        for key, value in result.scores.items():
            if _finite(value):
                p = p.field(key, float(value))
        points.append(p)
        for col in result.columns:
            if _finite(col["value"]):
                points.append(Point(MEASUREMENT_COLUMN).tag("dataset", result.dataset)
                              .tag("column", col["column"]).tag("kind", col["kind"])
                              .field(col["metric"], float(col["value"])))
        self.write_api.write(bucket=self.bucket, org=self.org, record=points)

    def close(self):
        self.client.close()
