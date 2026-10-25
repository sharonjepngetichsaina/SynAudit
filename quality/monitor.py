"""Sliding-window quality monitor, independent of Kafka.

Records are pushed in one at a time with add("real" | "synthetic", record). Every
`step` synthetic records, once both windows hold `min_fill` records, the monitor
scores the current windows and returns a WindowResult. Because it has no Kafka
dependency, the same class is used by the live consumer and by the CSV replay mode.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from . import metrics as m
from .clustering import RangeScaler, StreamClusterer

log = logging.getLogger("synaudit.monitor")

# Starting parameters per dataset (features are scaled to [0, 1]).
CLUSTER_DEFAULTS = {
    "adult": {"clustream": {"max_micro": 40}, "denstream": {"eps": 0.12, "mu": 8.0, "beta": 0.3, "lam": 0.002}},
    "credit-default": {"clustream": {"max_micro": 60}, "denstream": {"eps": 0.30, "mu": 8.0, "beta": 0.3, "lam": 0.002}},
}


@dataclass
class WindowResult:
    dataset: str
    window_id: int
    timestamp: str
    n_real: int
    n_synth: int
    scores: dict
    columns: list = field(default_factory=list)

    def flat(self):
        return {"dataset": self.dataset, "window_id": self.window_id, "timestamp": self.timestamp,
                "n_real": self.n_real, "n_synth": self.n_synth, **self.scores}


class QualityMonitor:
    def __init__(self, dataset, ctypes: m.ColumnTypes, window=500, step=100, min_fill=None,
                 sdmetrics_every=0, clustering=True, k_macro=5, cluster_params=None):
        self.dataset = dataset
        self.ctypes = ctypes
        self.window = window
        self.step = step
        self.min_fill = min_fill or window
        self.sdmetrics_every = sdmetrics_every   # 0 = never, N = every N-th window
        self.real = deque(maxlen=window)
        self.synth = deque(maxlen=window)
        self.since_emit = 0
        self.window_id = 0
        self.skipped = 0

        # all (real) value pairs seen so far, for the consistency check
        self.valid_pairs = {pair: set() for pair in m.CONSISTENCY_PAIRS.get(dataset, [])}

        self.clusterer = None
        if clustering:
            scaler = RangeScaler(ctypes.columns, ctypes.feature_ranges)
            params = cluster_params or CLUSTER_DEFAULTS.get(dataset, {})
            if scaler.dim == 0:
                log.warning("no numeric columns with ranges; clustering disabled")
            else:
                self.clusterer = StreamClusterer(scaler, k_macro=k_macro, **params)

    # ------------------------------------------------------------------ input
    def add(self, source, record):
        """Feed one record. Returns a WindowResult when a new window is ready, else None."""
        if source == "real":
            self.real.append(record)
            for (a, b), seen in self.valid_pairs.items():
                if record.get(a) is not None and record.get(b) is not None:
                    try:
                        seen.add((str(record[a]), float(record[b])))
                    except (TypeError, ValueError):
                        pass
        elif source == "synthetic":
            self.synth.append(record)
            self.since_emit += 1
        else:
            raise ValueError(f"unknown source {source!r}")

        if self.clusterer is not None:
            self.clusterer.add(source, record)

        if (source == "synthetic" and self.since_emit >= self.step
                and len(self.real) >= self.min_fill and len(self.synth) >= self.min_fill):
            self.since_emit = 0
            return self._score()
        return None

    # ---------------------------------------------------------------- scoring
    def _score(self):
        self.window_id += 1
        real_df = pd.DataFrame(list(self.real))
        synth_df = pd.DataFrame(list(self.synth))
        with_sd = bool(self.sdmetrics_every) and self.window_id % self.sdmetrics_every == 0

        summary, per_column = m.score_frames(real_df, synth_df, self.ctypes, self.dataset,
                                             valid_pairs=self.valid_pairs, with_sdmetrics=with_sd)
        if self.clusterer is not None:
            summary.update(self.clusterer.compare(m.coerce_frame(real_df, self.ctypes),
                                                  m.coerce_frame(synth_df, self.ctypes)))
        else:
            summary.update(StreamClusterer.empty_scores())

        return WindowResult(
            dataset=self.dataset, window_id=self.window_id,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            n_real=len(real_df), n_synth=len(synth_df), scores=summary, columns=per_column)


def format_line(result: WindowResult):
    s = result.scores

    def f(key):
        v = s.get(key, math.nan)
        return "  n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:5.3f}"

    return (f"[{result.dataset}] window {result.window_id:>4} | smd_max {f('smd_max')} tvd_mean {f('tvd_mean')} "
            f"corr {f('corr_diff_mean')} zero_gap {f('zero_gap_max')} consistency {f('consistency_min')} | "
            f"sdm {f('sdmetrics_overall')} | clu_cov {f('clustream_coverage_ratio')} "
            f"den_cov {f('denstream_coverage_ratio')} den_macro {f('denstream_macro_real')}/{f('denstream_macro_synth')}")
