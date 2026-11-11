"""Bloom filter

Answers "have I seen this item before?" using a fixed array of bits.
"No" is always correct. "Yes" is wrong with a small, chosen probability.

Reference:
  B. H. Bloom, "Space/Time Trade-offs in Hash Coding with Allowable Errors",
  Communications of the ACM, 13(7), 1970, pp. 422-426.
"""

import hashlib
import math


class BloomFilter:
    def __init__(self, expected_items, false_positive_rate=0.01):
        # Sizes from the standard formulas:
        #   m = -n * ln(p) / (ln 2)^2   bits
        #   k = (m / n) * ln 2          hash functions
        n = expected_items
        p = false_positive_rate
        self.m = math.ceil(-n * math.log(p) / (math.log(2) ** 2))
        self.k = max(1, round(self.m / n * math.log(2)))
        self.bits = bytearray(math.ceil(self.m / 8))
        self.count = 0

    def _positions(self, item):
        """The k bit positions for one item (double hashing)."""
        digest = hashlib.sha256(item.encode("utf-8")).digest()
        h1 = int.from_bytes(digest[:8], "big")
        h2 = int.from_bytes(digest[8:16], "big")
        return [(h1 + i * h2) % self.m for i in range(self.k)]

    def add(self, item):
        for pos in self._positions(item):
            self.bits[pos // 8] |= 1 << (pos % 8)
        self.count += 1

    def __contains__(self, item):
        return all(self.bits[pos // 8] & (1 << (pos % 8)) for pos in self._positions(item))