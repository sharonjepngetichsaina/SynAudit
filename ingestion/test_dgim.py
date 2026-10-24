"""Tests for the DGIM counter.

Two matter most: that the structural invariants hold after every arrival, and
that the observed error stays inside the bound the paper proves.

Run:  python ingestion/test_dgim.py
"""

import random
from collections import Counter

from dgim import DGIMCounter


def assert_invariants(d: DGIMCounter) -> None:
    sizes = [b.size for b in d.buckets]
    timestamps = [b.timestamp for b in d.buckets]

    assert all(s & (s - 1) == 0 for s in sizes), f"sizes not powers of two: {sizes}"
    assert sizes == sorted(sizes), f"sizes not non-decreasing with age: {sizes}"
    assert timestamps == sorted(timestamps, reverse=True), "buckets not newest-first"

    counts = Counter(sizes)
    too_many = {s: n for s, n in counts.items() if n > d.buckets_per_size}
    assert not too_many, f"too many buckets of a size: {too_many}"


def test_invariants_hold_throughout() -> None:
    random.seed(2)
    d = DGIMCounter(window=256)
    for _ in range(20000):
        d.add(1 if random.random() < 0.4 else 0)
        assert_invariants(d)
    print("pass: invariants hold across 20,000 arrivals")


def test_empty_and_all_zeros() -> None:
    d = DGIMCounter(window=10)
    assert d.count() == 0
    for _ in range(10):
        d.add(0)
    assert d.count() == 0
    print("pass: empty and all-zero streams count 0")


def test_all_ones_stays_within_bound() -> None:
    """With every element a 1 the true count is the window size itself.

    The estimate is not exact even here: how close it lands depends on the
    bucket configuration at that moment, so window 4096 reads 3072. What must
    hold is that it never exceeds the window and stays inside the error bound.
    """
    for window in (5, 100, 1000, 4096):
        for multiplier in (3, 20):
            d = DGIMCounter(window=window)
            for _ in range(window * multiplier):
                d.add(1)

            estimate = d.count()
            assert estimate <= window, (
                f"window {window}: estimate {estimate} exceeds the window"
            )
            error = abs(estimate - window) / window
            assert error <= d.error_bound, (
                f"window {window}: error {error:.2f} exceeds bound {d.error_bound:.2f}"
            )
    print("pass: all-ones stream stays within the window and the error bound")


def test_count_never_exceeds_window() -> None:
    """The window holds N elements, so no estimate may exceed N — a rate above
    1.0 is impossible and would be visible nonsense on a dashboard."""
    random.seed(9)
    for density in (0.5, 0.73, 0.95, 1.0):
        d = DGIMCounter(window=500)
        for _ in range(5000):
            d.add(1 if random.random() < density else 0)
            assert d.count() <= 500, f"density {density}: estimate {d.count()} > window 500"
    print("pass: estimate never exceeds the window size")


def test_error_within_bound(window: int = 1000, trials: int = 20000) -> None:
    """Observed relative error must stay inside 1/(buckets_per_size - 1)."""
    random.seed(1)
    stream = [1 if random.random() < 0.5 else 0 for _ in range(trials)]

    for r in (2, 3, 4):
        d = DGIMCounter(window=window, buckets_per_size=r)
        errors = []
        for i, bit in enumerate(stream):
            d.add(bit)
            if i + 1 >= window:
                true = sum(stream[i + 1 - window : i + 1])
                if true:
                    errors.append(abs(d.count() - true) / true)

        worst = max(errors)
        bound = d.error_bound
        assert worst <= bound, f"r={r}: error {worst:.3f} exceeds bound {bound:.3f}"
        print(f"pass: r={r}  max error {worst:.4f} <= bound {bound:.3f}  "
              f"({d.memory_buckets} buckets)")


def test_memory_is_sublinear() -> None:
    """Bucket count grows logarithmically, not linearly, in the window."""
    counts = []
    for window in (64, 1024, 16384):
        d = DGIMCounter(window=window)
        for _ in range(window * 2):
            d.add(1)
        counts.append((window, d.memory_buckets))

    # A 256x larger window must not need anything close to 256x the buckets.
    (small_w, small_b), (_, _), (large_w, large_b) = counts
    ratio = large_b / small_b
    assert ratio < 10, f"memory grew {ratio:.1f}x for a {large_w // small_w}x window"
    for window, buckets in counts:
        print(f"pass: window {window:6,} -> {buckets:3d} buckets")


def test_rejects_invalid_parameters() -> None:
    for bad_window in (0, -5):
        try:
            DGIMCounter(window=bad_window)
        except ValueError:
            continue
        raise AssertionError(f"window={bad_window} should have been rejected")

    try:
        DGIMCounter(window=10, buckets_per_size=1)
    except ValueError:
        print("pass: invalid parameters rejected")
        return
    raise AssertionError("buckets_per_size=1 should have been rejected")


if __name__ == "__main__":
    test_empty_and_all_zeros()
    test_all_ones_stays_within_bound()
    test_count_never_exceeds_window()
    test_rejects_invalid_parameters()
    test_invariants_hold_throughout()
    test_memory_is_sublinear()
    test_error_within_bound()
    print("\nall tests passed")
