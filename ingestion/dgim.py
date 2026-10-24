"""DGIM: counting 1s in a sliding window over a binary stream.

Answers "how many 1s in the last N elements" in O(log^2 N) bits instead of the
N bits an exact window needs, with bounded relative error.

Implemented from the paper rather than taken from a library: this is a course
requirement, and the error bound is the part that has to be defensible.

Reference:
  M. Datar, A. Gionis, P. Indyk, R. Motwani,
  "Maintaining Stream Statistics over Sliding Windows",
  SIAM Journal on Computing, 31(6), 2002, pp. 1794-1813.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Bucket:
    """A group of 1s. `timestamp` is the position of its most recent 1,
    `size` is how many 1s it holds — always a power of two."""

    timestamp: int
    size: int


class DGIMCounter:
    """Approximate count of 1s in the last `window` elements of a binary stream.

    Invariants maintained after every arrival:
      1. every bucket size is a power of two
      2. at most `buckets_per_size` buckets share any one size
      3. sizes are non-decreasing going back in time
      4. buckets partition the 1s in the window

    `buckets_per_size` trades memory for accuracy: the relative error is bounded
    by roughly 1/(buckets_per_size - 1), so the default of 2 gives the standard
    50% bound and larger values tighten it.
    """

    def __init__(self, window: int, buckets_per_size: int = 2) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        if buckets_per_size < 2:
            raise ValueError("buckets_per_size must be at least 2")

        self.window = window
        self.buckets_per_size = buckets_per_size
        self.position = 0                  # elements seen, ever
        self.buckets: list[Bucket] = []    # newest first

    def add(self, bit: int) -> None:
        """Offer one element of the stream. Amortised O(1)."""
        self.position += 1
        self._expire()

        if not bit:
            return

        # A new 1 starts its own bucket of size 1 at the front.
        self.buckets.insert(0, Bucket(timestamp=self.position, size=1))
        self._merge()

    def _expire(self) -> None:
        """Drop buckets whose most recent 1 has left the window.

        Buckets are ordered newest first, so the stale ones are always at the
        tail and can be cut in one pass.
        """
        cutoff = self.position - self.window
        while self.buckets and self.buckets[-1].timestamp <= cutoff:
            self.buckets.pop()

    def _merge(self) -> None:
        """Restore invariant 2 by merging the two oldest buckets of any size
        that now has too many, cascading upward.

        Merging the two *oldest* is what keeps invariant 3: the combined bucket
        is larger and sits further back, so sizes stay non-decreasing with age.
        """
        i = 0
        while i < len(self.buckets):
            size = self.buckets[i].size

            # Buckets of equal size are contiguous, so count the run.
            run_start = i
            while i < len(self.buckets) and self.buckets[i].size == size:
                i += 1
            run_end = i  # exclusive

            if run_end - run_start <= self.buckets_per_size:
                continue  # this size is within budget; move to the next

            # Too many of this size: merge the two oldest of the run. The merged
            # bucket keeps the newer of the two timestamps, because that is the
            # most recent 1 it now contains.
            older = self.buckets[run_end - 1]
            newer = self.buckets[run_end - 2]
            merged = Bucket(timestamp=newer.timestamp, size=size * 2)

            self.buckets[run_end - 2 : run_end] = [merged]

            # The merge may have created too many of the doubled size, so
            # continue the scan from the merged bucket rather than past it.
            i = run_end - 2

    def count(self) -> int:
        """Estimated number of 1s in the last `window` elements.

        Every bucket but the oldest lies entirely inside the window and is
        counted in full. The oldest straddles the window edge: its 1s are known
        but how many remain in the window is not, so half is taken. That single
        half-bucket is the only source of error.
        """
        if not self.buckets:
            return 0

        total = sum(b.size for b in self.buckets[:-1])
        total += self.buckets[-1].size // 2 or 1  # a size-1 bucket contributes 1

        # The oldest bucket's 1s may lie mostly outside the window, so taking
        # half of it can overshoot: an all-ones stream with N=1000 estimates
        # 1160. The window holds N elements, so the count cannot exceed N.
        # Clamping costs nothing and strictly reduces error.
        return min(total, self.window)

    @property
    def memory_buckets(self) -> int:
        """Number of buckets held — the structure's actual memory cost."""
        return len(self.buckets)

    @property
    def error_bound(self) -> float:
        """Worst-case relative error, 1/(buckets_per_size - 1)."""
        return 1.0 / (self.buckets_per_size - 1)

    def __repr__(self) -> str:
        return (
            f"DGIMCounter(window={self.window}, position={self.position}, "
            f"estimate={self.count()}, buckets={len(self.buckets)})"
        )
