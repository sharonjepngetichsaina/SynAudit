"""HyperLogLog (Flajolet, Fusy, Gandouet and Meunier, 2007).

Estimates the number of distinct items in a stream with m = 2^p small registers.

  * the first p bits of the item's 64-bit hash choose a register j
  * rho = position of the first 1-bit in the remaining bits
  * register[j] keeps the largest rho seen
  * estimate = alpha_m * m^2 / sum(2^-register[j])

The standard error is about 1.04 / sqrt(m). For small cardinalities, where many
registers are still empty, the paper switches to linear counting.
"""
from __future__ import annotations

import math

from sketches.hashing import hash64


def _alpha(m: int) -> float:
    if m == 16:
        return 0.673
    if m == 32:
        return 0.697
    if m == 64:
        return 0.709
    return 0.7213 / (1 + 1.079 / m)


class HyperLogLog:
    def __init__(self, precision: int = 10):
        if not 4 <= precision <= 16:
            raise ValueError("precision must be between 4 and 16")
        self.p = precision
        self.m = 1 << precision
        self.registers = [0] * self.m

    def update(self, item) -> None:
        x = hash64(item, seed=1_000)
        j = x >> (64 - self.p)                       # first p bits pick the register
        rest = x & ((1 << (64 - self.p)) - 1)        # remaining 64 - p bits
        rho = (64 - self.p) - rest.bit_length() + 1  # position of the first 1-bit
        if rho > self.registers[j]:
            self.registers[j] = rho

    def estimate(self) -> float:
        raw = _alpha(self.m) * self.m ** 2 / sum(2.0 ** -r for r in self.registers)
        zeros = self.registers.count(0)
        if raw <= 2.5 * self.m and zeros:            # small-range correction
            return self.m * math.log(self.m / zeros)
        return raw

    @property
    def standard_error(self) -> float:
        return 1.04 / math.sqrt(self.m)

    @property
    def memory_bytes(self) -> int:
        return self.m                                 # one byte per register
