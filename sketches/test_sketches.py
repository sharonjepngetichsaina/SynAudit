"""Checks that the sketches keep the guarantees from the papers.

    python -m sketches.test_sketches        (or: pytest sketches/test_sketches.py)
"""
import random
from collections import Counter

from sketches.count_min import CountMinSketch
from sketches.hyperloglog import HyperLogLog


def zipf_stream(n, k, seed=0):
    rng = random.Random(seed)
    weights = [1 / (i + 1) for i in range(k)]
    return rng.choices([f"item{i}" for i in range(k)], weights=weights, k=n)


def test_cms_never_undercounts():
    cms, exact = CountMinSketch(50, 4), Counter()
    for x in zipf_stream(20_000, 500):
        cms.update(x); exact[x] += 1
    assert all(cms.estimate(k) >= v for k, v in exact.items())


def test_cms_error_bound():
    eps, delta, n = 0.01, 0.01, 20_000
    cms, exact = CountMinSketch.from_error(eps, delta), Counter()
    for x in zipf_stream(n, 2_000, seed=1):
        cms.update(x); exact[x] += 1
    over = [cms.estimate(k) - v for k, v in exact.items()]
    within = sum(e <= eps * n for e in over) / len(over)
    assert within >= 1 - delta, within        # holds for at least 1 - delta of the items


def test_hll_accuracy():
    for true_n in (50, 1_000, 50_000):
        hll = HyperLogLog(precision=10)
        for i in range(true_n):
            hll.update(f"user{i}")
        err = abs(hll.estimate() - true_n) / true_n
        assert err < 4 * hll.standard_error, (true_n, err)


def test_hll_ignores_duplicates():
    hll = HyperLogLog(10)
    for _ in range(100):
        for i in range(200):
            hll.update(i)
    assert abs(hll.estimate() - 200) / 200 < 0.1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
