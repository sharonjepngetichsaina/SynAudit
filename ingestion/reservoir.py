"""Reservoir sampling (Vitter 1985, Algorithm R).

Maintains a uniform random sample of fixed size k over a stream of unknown
length, in one pass and O(k) memory. At any point, every record seen so far is
in the reservoir with probability k/n.

Implemented from the paper rather than taken from a library: this is a course
requirement, and the invariant is the part that has to be defensible.

Reference:
  J. S. Vitter, "Random Sampling with a Reservoir",
  ACM Transactions on Mathematical Software, 11(1), 1985, pp. 37-57.
"""

from __future__ import annotations

import random
from typing import Any


class ReservoirSampler:
    """Uniform sample of size k over a stream of unknown length.

    The reservoir holds the first k items outright. From item k+1 onwards each
    arrival either replaces a uniformly chosen slot or is discarded, with the
    acceptance probability falling as k/i so that the sample stays uniform.

    Seeded explicitly so that a run can be reproduced exactly, which matters
    when a sampled reference set feeds the quality and privacy comparisons.
    """

    def __init__(self, k: int, seed: int | None = None) -> None:
        if k <= 0:
            raise ValueError("reservoir size k must be positive")
        self.k = k
        self.seen = 0                      # n: items seen so far
        self.reservoir: list[Any] = []
        self._rng = random.Random(seed)

    def add(self, item: Any) -> None:
        """Offer one item to the reservoir. O(1)."""
        self.seen += 1

        if len(self.reservoir) < self.k:
            # Filling phase: the first k items are kept unconditionally.
            self.reservoir.append(item)
            return

        # Replacement phase. j is uniform over {1..seen}; the item is accepted
        # with probability k/seen, and then displaces a uniformly chosen slot.
        j = self._rng.randint(1, self.seen)
        if j <= self.k:
            self.reservoir[j - 1] = item

    def sample(self) -> list[Any]:
        """The current sample. A copy, so callers cannot mutate internal state."""
        return list(self.reservoir)

    @property
    def is_full(self) -> bool:
        return len(self.reservoir) >= self.k

    @property
    def inclusion_probability(self) -> float:
        """k/n: the chance any one seen record is currently in the reservoir."""
        if self.seen == 0:
            return 0.0
        return min(1.0, self.k / self.seen)

    def __len__(self) -> int:
        return len(self.reservoir)

    def __repr__(self) -> str:
        return (
            f"ReservoirSampler(k={self.k}, seen={self.seen}, "
            f"held={len(self.reservoir)}, p={self.inclusion_probability:.4f})"
        )
