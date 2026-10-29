# ML utility evaluation

Measures whether synthetic data is useful for machine learning, using
train-on-synthetic, test-on-real (TSTR) compared with train-on-real,
test-on-real (TRTR).

## Algorithm
**Hoeffding tree** (Domingos and Hulten, 2000, "Mining High-Speed Data Streams", Proc. KDD 2000, pp. 71-80). A decision tree that learns from one record at a time, so it never needs the whole dataset in memory. Implemented with  `river.tree.HoeffdingTreeClassifier`.

## How it works

1. 20% of the real records are held out as a test set that no model sees.
2. Model S learns from synthetic records, model R from real records.
3. Every 500 records both models are scored on the same 2,000 real test records.
4. Utility gap = balanced accuracy of R minus balanced accuracy of S.

Balanced accuracy and F1 are reported next to accuracy, because the classes
are imbalanced (about 24% `>50K` in Adult, 22% defaults in Credit).

## Running it

```bash
python utility/tstr.py --dataset adult
python utility/tstr.py --dataset credit-default
```

## Output

`results/utility/<dataset>-utility.csv`, one row per window:
`window`, `rows_seen`, accuracy, balanced accuracy and F1 for both models, and `utility_gap`.

## Results so far

| Dataset | Balanced accuracy R | Balanced accuracy S | Gap |
|---|---|---|---|
| Adult | about 0.74 | 0.50 | about 0.24 |
| Credit Default | about 0.64 | 0.50 | about 0.14 |

The model trained on synthetic data only predicts the majority class: the
current synthetic data does not keep the link between the features and the label.