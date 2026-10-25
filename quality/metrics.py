"""Quality metrics shared by the batch evaluator and the streaming monitor.

Everything here works on two pandas DataFrames (a real sample and a synthetic
sample), so the same code scores a whole CSV or one sliding window of a stream.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("synaudit.metrics")

# Pairs (categorical, numeric) that must always agree row by row.
CONSISTENCY_PAIRS = {
    "adult": [("education", "education_num")],
    "credit-default": [],
}

# A numeric column is "zero-inflated" when at least this share of real values is 0.
ZERO_INFLATED_MIN_SHARE = 0.5

MISSING = "<MISSING>"


@dataclass
class ColumnTypes:
    numeric: list[str]
    categorical: list[str]
    # numeric-like columns usable as clustering features, with (min, max) ranges
    feature_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def columns(self) -> list[str]:
        return self.numeric + self.categorical

    def sdmetrics_metadata(self) -> dict:
        cols = {c: {"sdtype": "numerical"} for c in self.numeric}
        cols.update({c: {"sdtype": "categorical"} for c in self.categorical})
        return {"columns": cols}


def load_column_types(metadata_path: str | Path) -> ColumnTypes:
    """Read column types from the SDV metadata JSON written by generate_synthetic.py."""
    meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    table = next(iter(meta["tables"].values()))
    numeric, categorical, ranges = [], [], {}
    for name, spec in table["columns"].items():
        sdtype = spec.get("sdtype")
        if sdtype == "numerical":
            numeric.append(name)
            lo, hi = spec.get("range_min"), spec.get("range_max")
            if lo is not None and hi is not None and hi > lo:
                ranges[name] = (float(lo), float(hi))
        else:
            categorical.append(name)
            values = spec.get("range_values") or []
            if values and all(isinstance(v, (int, float)) for v in values):
                lo, hi = float(min(values)), float(max(values))
                if hi > lo:
                    ranges[name] = (lo, hi)
    return ColumnTypes(numeric, categorical, ranges)


def infer_column_types(df: pd.DataFrame) -> ColumnTypes:
    """Fallback when no metadata file exists: use pandas dtypes."""
    numeric = list(df.select_dtypes(include="number").columns)
    categorical = [c for c in df.columns if c not in numeric]
    ranges = {}
    for c in numeric:
        lo, hi = float(df[c].min()), float(df[c].max())
        if hi > lo:
            ranges[c] = (lo, hi)
    return ColumnTypes(numeric, categorical, ranges)


def coerce_frame(df: pd.DataFrame, ctypes: ColumnTypes) -> pd.DataFrame:
    """Make Kafka/JSON values comparable: numbers as numbers, the rest as text."""
    out = pd.DataFrame(index=df.index)
    for c in ctypes.numeric:
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    for c in ctypes.categorical:
        if c in df:
            out[c] = df[c].where(df[c].notna(), MISSING).astype(str)
        else:
            out[c] = MISSING
    return out[ctypes.columns]


# --------------------------------------------------------------------- metrics
def smd_per_column(real, synth, cols) -> dict[str, float]:
    """|mean_synth - mean_real| / std_real (0 is perfect, >0.5 is a clear shift)."""
    out = {}
    for c in cols:
        r = pd.to_numeric(real[c], errors="coerce").dropna()
        s = pd.to_numeric(synth[c], errors="coerce").dropna()
        if r.empty or s.empty:
            out[c] = math.nan
            continue
        sd = r.std()
        out[c] = abs(s.mean() - r.mean()) / sd if pd.notna(sd) and sd > 0 else math.nan
    return out


def tvd_per_column(real, synth, cols) -> dict[str, float]:
    """Total variation distance between category proportions (0 identical, 1 disjoint)."""
    out = {}
    for c in cols:
        rf = real[c].fillna(MISSING).astype(str).value_counts(normalize=True)
        sf = synth[c].fillna(MISSING).astype(str).value_counts(normalize=True)
        cats = rf.index.union(sf.index)
        out[c] = float(0.5 * (rf.reindex(cats, fill_value=0) - sf.reindex(cats, fill_value=0)).abs().sum())
    return out


def corr_difference(real, synth, cols) -> pd.DataFrame | None:
    if len(cols) < 2:
        return None
    return (real[cols].corr() - synth[cols].corr()).abs()


def mean_corr_difference(real, synth, cols) -> float:
    diff = corr_difference(real, synth, cols)
    if diff is None:
        return math.nan
    return float(np.nanmean(diff.to_numpy()))


def zero_shares(df, cols) -> dict[str, float]:
    return {c: float((pd.to_numeric(df[c], errors="coerce") == 0).mean()) for c in cols}


def zero_share_gap(real, synth, cols, min_share=ZERO_INFLATED_MIN_SHARE) -> dict[str, float]:
    """For zero-inflated columns: |share of zeros (real) - share of zeros (synthetic)|.

    This catches the capital_gain problem: a Gaussian copula smears the 92% zero
    spike into a continuous distribution, so the gap becomes ~0.9.
    """
    rz, sz = zero_shares(real, cols), zero_shares(synth, cols)
    return {c: abs(rz[c] - sz[c]) for c in cols if rz[c] >= min_share}


def learn_valid_pairs(real, a: str, b: str) -> set:
    """Set of (a, b) value pairs that occur in real data."""
    sub = real[[a, b]].dropna()
    return set(zip(sub[a].astype(str), sub[b].astype(float)))


def consistency_rate(synth, valid_pairs: set, a: str, b: str) -> float:
    """Share of synthetic rows whose (a, b) pair also occurs in the real data.

    1.0 means every synthetic row is a combination that exists in reality
    (e.g. HS-grad always comes with education_num 9).
    """
    if not valid_pairs or synth.empty:
        return math.nan
    sub = synth[[a, b]].dropna()
    if sub.empty:
        return math.nan
    ok = [(str(x), float(y)) in valid_pairs for x, y in zip(sub[a], sub[b])]
    return float(np.mean(ok))


def sdmetrics_scores(real, synth, ctypes: ColumnTypes) -> dict[str, float]:
    """Official SDMetrics Quality Report: Column Shapes and Column Pair Trends."""
    nan = {"sdmetrics_overall": math.nan, "sdmetrics_column_shapes": math.nan,
           "sdmetrics_column_pair_trends": math.nan}
    try:
        from sdmetrics.reports.single_table import QualityReport
    except ImportError:
        log.warning("sdmetrics is not installed; skipping SDMetrics scores")
        return nan
    try:
        report = QualityReport()
        report.generate(real, synth, ctypes.sdmetrics_metadata(), verbose=False)
        props = report.get_properties()
        by_name = dict(zip(props["Property"], props["Score"]))
        return {
            "sdmetrics_overall": float(report.get_score()),
            "sdmetrics_column_shapes": float(by_name.get("Column Shapes", math.nan)),
            "sdmetrics_column_pair_trends": float(by_name.get("Column Pair Trends", math.nan)),
        }
    except Exception as exc:  # window too small, constant column, version differences
        log.warning("SDMetrics failed on this sample: %s", exc)
        return nan


# --------------------------------------------------------------- one-shot score
def score_frames(real, synth, ctypes: ColumnTypes, dataset: str, valid_pairs: dict | None = None,
                 with_sdmetrics: bool = False):
    """Score a real sample against a synthetic sample.

    Returns (summary, per_column):
      summary     flat dict of floats (one row per window / dataset)
      per_column  list of dicts {column, kind, metric, value}
    """
    real = coerce_frame(real, ctypes)
    synth = coerce_frame(synth, ctypes)

    smd = smd_per_column(real, synth, ctypes.numeric)
    tvd = tvd_per_column(real, synth, ctypes.categorical)
    zero_gap = zero_share_gap(real, synth, ctypes.numeric)

    def agg(d, fn):
        vals = [v for v in d.values() if v is not None and not math.isnan(v)]
        return float(fn(vals)) if vals else math.nan

    summary = {
        "smd_mean": agg(smd, np.mean),
        "smd_max": agg(smd, np.max),
        "tvd_mean": agg(tvd, np.mean),
        "tvd_max": agg(tvd, np.max),
        "corr_diff_mean": mean_corr_difference(real, synth, ctypes.numeric),
        "zero_gap_max": agg(zero_gap, np.max),
    }

    rates = []
    for a, b in CONSISTENCY_PAIRS.get(dataset, []):
        if a in real and b in real:
            pairs = (valid_pairs or {}).get((a, b))
            if pairs is None:
                pairs = learn_valid_pairs(real, a, b)
            rate = consistency_rate(synth, pairs, a, b)
            summary[f"consistency_{a}_{b}"] = rate
            rates.append(rate)
    summary["consistency_min"] = agg({i: r for i, r in enumerate(rates)}, np.min) if rates else math.nan

    summary.update(sdmetrics_scores(real, synth, ctypes) if with_sdmetrics else
                   {"sdmetrics_overall": math.nan, "sdmetrics_column_shapes": math.nan,
                    "sdmetrics_column_pair_trends": math.nan})

    per_column = (
        [{"column": c, "kind": "numeric", "metric": "smd", "value": v} for c, v in smd.items()]
        + [{"column": c, "kind": "categorical", "metric": "tvd", "value": v} for c, v in tvd.items()]
        + [{"column": c, "kind": "numeric", "metric": "zero_gap", "value": v} for c, v in zero_gap.items()]
    )
    return summary, per_column
