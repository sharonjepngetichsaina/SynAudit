"""Sketches for every categorical column of one stream, plus exact counts to measure their error."""
from __future__ import annotations

import sys
from collections import Counter

from sketches.count_min import CountMinSketch
from sketches.hyperloglog import HyperLogLog


class ColumnSketches:
    def __init__(self, cms_width: int, cms_depth: int, hll_precision: int, exact: bool = True):
        self.cms = CountMinSketch(cms_width, cms_depth)
        self.hll = HyperLogLog(hll_precision)
        self.exact = Counter() if exact else None   # ground truth, only to evaluate the sketches

    def update(self, value) -> None:
        self.cms.update(value)
        self.hll.update(value)
        if self.exact is not None:
            self.exact[value] += 1


class StreamSketches:
    """All categorical columns of one (dataset, source) stream."""

    def __init__(self, cms_width=272, cms_depth=5, hll_precision=10, exact=True):
        self.args = (cms_width, cms_depth, hll_precision, exact)
        self.columns: dict[str, ColumnSketches] = {}
        self.n = 0

    def update(self, payload: dict) -> None:
        self.n += 1
        for col, value in payload.items():
            if isinstance(value, str):                # categorical columns are JSON strings (stream contract)
                self.columns.setdefault(col, ColumnSketches(*self.args)).update(value)

    def summary(self, column: str) -> dict:
        """Distinct count (HLL) and frequency-error (CMS) for one column, against exact counts."""
        c = self.columns[column]
        out = {"records_seen": self.n,
               "hll_estimate": c.hll.estimate(), "hll_memory_bytes": c.hll.memory_bytes,
               "cms_memory_bytes": c.cms.memory_bytes, "cms_epsilon": c.cms.epsilon}
        if c.exact is not None and c.exact:
            distinct = len(c.exact)
            errors = [c.cms.estimate(k) - v for k, v in c.exact.items()]
            out.update(hll_exact=distinct,
                       hll_rel_error=abs(out["hll_estimate"] - distinct) / distinct,
                       cms_max_error_share=max(errors) / c.cms.n,
                       cms_mean_error_share=sum(errors) / len(errors) / c.cms.n,
                       exact_memory_bytes=sys.getsizeof(c.exact) + sum(sys.getsizeof(k) for k in c.exact))
        return out

    def shares(self, column: str, categories) -> dict:
        """Estimated share of each category in this stream (CMS estimate / records)."""
        c = self.columns.get(column)
        if c is None or c.cms.n == 0:
            return {}
        return {k: c.cms.estimate(k) / c.cms.n for k in categories}
