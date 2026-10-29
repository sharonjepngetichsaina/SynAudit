# Privacy risk evaluation

Checks whether synthetic records are exact copies of real people.

## Algorithm

**Bloom filter**: Implemented from the paper in `bloom.py`. Size is set by
- bits: m = -n ln(p) / (ln 2)^2
- hash functions: k = (m / n) ln 2

A "no" answer is always correct; a "yes" is wrong with probability about p (1%).

## Files

- `bloom.py`: the Bloom filter
- `test_bloom.py`: shows no missed items and about 1% false positives
- `copy_check.py`: adds every real record to the filter, then streams the
  synthetic records and reports the copy rate per window of 500 records

## Running it

```bash
python privacy/test_bloom.py
python privacy/copy_check.py --dataset adult
python privacy/copy_check.py --dataset credit-default
```

## Output

`results/privacy/<dataset>-copies.csv`, one row per window:
`window`, `rows_seen`, `bloom_copy_rate`, `exact_copy_rate`.

## Results so far

| Dataset | Real rows repeating real rows | Synthetic exact copies | Bloom filter reports |
|---|---|---|---|
| Adult | 13.1% | 0.36% | 0.78% |
| Credit Default | 0.3% | 0% | 0.82% |

The Bloom filter never misses a copy; its extra flags are its ~1% false positives.