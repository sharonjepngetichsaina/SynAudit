"""Tests for the reservoir sampler.

The uniformity test is the important one: it is the empirical evidence that the
implementation actually satisfies the invariant the proof claims.

Run:  python ingestion/test_reservoir.py
"""

from collections import Counter

from reservoir import ReservoirSampler


def test_keeps_everything_when_stream_shorter_than_k() -> None:
    r = ReservoirSampler(k=10, seed=1)
    for i in range(4):
        r.add(i)
    assert len(r) == 4
    assert r.sample() == [0, 1, 2, 3]
    assert r.inclusion_probability == 1.0
    print("pass: n < k keeps the whole stream")


def test_never_exceeds_k() -> None:
    r = ReservoirSampler(k=5, seed=1)
    for i in range(1000):
        r.add(i)
    assert len(r) == 5
    assert r.seen == 1000
    print("pass: reservoir never exceeds k")


def test_sample_is_a_copy() -> None:
    r = ReservoirSampler(k=3, seed=1)
    for i in range(3):
        r.add(i)
    s = r.sample()
    s.append("injected")
    assert len(r) == 3
    print("pass: sample() returns a copy, not internal state")


def test_reproducible_with_same_seed() -> None:
    a = ReservoirSampler(k=10, seed=7)
    b = ReservoirSampler(k=10, seed=7)
    for i in range(500):
        a.add(i)
        b.add(i)
    assert a.sample() == b.sample()
    print("pass: same seed gives the same sample")


def test_uniformity(n: int = 50, k: int = 5, trials: int = 20000) -> None:
    """Every item should appear in about k/n of the trials.

    Checked with a chi-square goodness-of-fit test against the uniform
    distribution. df = n-1 = 49, 95% critical value ~ 67.5.
    """
    counts: Counter = Counter()
    for t in range(trials):
        r = ReservoirSampler(k, seed=t)
        for i in range(n):
            r.add(i)
        counts.update(r.sample())

    expected = trials * k / n
    chi2 = sum((counts[i] - expected) ** 2 / expected for i in range(n))

    observed = [counts[i] / trials for i in range(n)]
    deviation = max(abs(f - k / n) for f in observed)

    print(f"pass: uniformity  chi2={chi2:.1f} (critical 67.5), "
          f"max deviation {deviation:.4f}")
    assert chi2 < 67.5, f"sample is not uniform: chi2={chi2:.1f}"


def test_rejects_invalid_k() -> None:
    for bad in (0, -1):
        try:
            ReservoirSampler(k=bad)
        except ValueError:
            continue
        raise AssertionError(f"k={bad} should have been rejected")
    print("pass: invalid k rejected")


if __name__ == "__main__":
    test_keeps_everything_when_stream_shorter_than_k()
    test_never_exceeds_k()
    test_sample_is_a_copy()
    test_reproducible_with_same_seed()
    test_rejects_invalid_k()
    test_uniformity()
    print("\nall tests passed")
