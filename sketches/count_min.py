"""Count-Min Sketch (Cormode and Muthukrishnan, 2005).

Estimates how often each item appears in a stream using a fixed d x w table of
counters, whatever the number of distinct items.

  * update(x): for each row i, add 1 to counter [i, h_i(x)]
  * estimate(x): the minimum of those d counters

Collisions only ever add to a counter, so the estimate never undercounts. With
w = ceil(e / epsilon) and d = ceil(ln(1 / delta)) the paper proves
    estimate(x) <= true(x) + epsilon * N    with probability at least 1 - delta,
where N is the number of updates so far.
"""
from __future__ import annotations

import math

from sketches.hashing import hash64


class CountMinSketch:
    def __init__(self, width: int, depth: int):
        if width < 1 or depth < 1:
            raise ValueError("width and depth must be positive")
        self.width, self.depth = width, depth
        self.table = [[0] * width for _ in range(depth)]
        self.n = 0

    @classmethod
    def from_error(cls, epsilon: float, delta: float) -> "CountMinSketch":
        """Size the table from the error guarantee, as in the paper."""
        return cls(math.ceil(math.e / epsilon), math.ceil(math.log(1 / delta)))

    def _cells(self, item):
        return [(row, hash64(item, seed=row) % self.width) for row in range(self.depth)]

    def update(self, item, count: int = 1) -> None:
        for row, col in self._cells(item):
            self.table[row][col] += count
        self.n += count

    def estimate(self, item) -> int:
        return min(self.table[row][col] for row, col in self._cells(item))

    @property
    def epsilon(self) -> float:
        return math.e / self.width

    @property
    def memory_bytes(self) -> int:
        return self.width * self.depth * 4      # 32-bit counters
