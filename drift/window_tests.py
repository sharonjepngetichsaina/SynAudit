"""Window-level drift tests: compare each window of the stream with the
reference sample using a two-sample statistical test per column.

Numeric columns:     two-sample Kolmogorov-Smirnov (KS) test.
Categorical columns: chi-square test of homogeneity.

Each window tests several columns, so the significance level is
Bonferroni-corrected: a column raises an alarm when p < alpha / n_tests.
"""

import numpy as np
import pandas as pd
from scipy import stats


def ks_test(ref_values, win_values):
    """Two-sample KS test on one numeric column. Missing values are dropped.

    Returns (statistic D, p-value). D is the largest gap between the two
    empirical cumulative distribution functions (0 = identical, 1 = no overlap).
    """
    ref_values = pd.Series(ref_values).dropna()
    win_values = pd.Series(win_values).dropna()
    if len(ref_values) == 0 or len(win_values) == 0:
        return 0.0, 1.0
    result = stats.ks_2samp(ref_values, win_values)
    return float(result.statistic), float(result.pvalue)


def merge_rare(ref_values, win_values, min_share=0.01):
    """Prepare two categorical samples for the chi-square test.

    Missing values become the category 'missing'. Categories with less than
    `min_share` of the reference become 'other': the chi-square test needs
    expected counts of about 5 or more, and in a window of 500 records a
    category with a 0.1 % share is expected only 0.5 times.
    """
    ref_values = pd.Series(ref_values).astype("object").fillna("missing")
    win_values = pd.Series(win_values).astype("object").fillna("missing")
    share = ref_values.value_counts(normalize=True)
    keep = set(share[share >= min_share].index)
    return (ref_values.where(ref_values.isin(keep), "other"),
            win_values.where(win_values.isin(keep), "other"))


def chi_square_test(ref_values, win_values, min_share=0.01):
    """Chi-square test of homogeneity on one categorical column.

    Builds a 2 x k contingency table (reference counts, window counts) and
    tests whether both rows have the same category proportions.
    Returns (chi-square statistic, p-value).
    """
    ref_values, win_values = merge_rare(ref_values, win_values, min_share)
    categories = sorted(set(ref_values) | set(win_values))
    table = np.array([
        ref_values.value_counts().reindex(categories, fill_value=0).to_numpy(),
        win_values.value_counts().reindex(categories, fill_value=0).to_numpy(),
    ])
    if table.shape[1] < 2:           # only one category: nothing to compare
        return 0.0, 1.0
    chi2, p_value, _, _ = stats.chi2_contingency(table)
    return float(chi2), float(p_value)


def window_tests(reference, stream, num_cols, cat_cols=(), window_size=500,
                 alpha=0.05, n_tests=None, dataset="adult"):
    """Run the KS test (numeric) and the chi-square test (categorical) for
    every column in every window of the stream.

    Returns one row per (window, column), shaped like the `drift_events`
    records we will write to InfluxDB. `n_tests` is the number of tests per
    window used for the Bonferroni correction (default: number of columns).
    """
    n_tests = n_tests or (len(num_cols) + len(cat_cols))
    threshold = alpha / n_tests
    rows = []
    for window_id, start in enumerate(range(0, len(stream) - window_size + 1, window_size)):
        window = stream.iloc[start:start + window_size]
        for col in num_cols:
            statistic, p_value = ks_test(reference[col], window[col])
            rows.append(dict(window_id=window_id, start=start, end=start + window_size,
                             dataset=dataset, column=col, detector="ks",
                             statistic=statistic, p_value=p_value,
                             alarm=int(p_value < threshold)))
        for col in cat_cols:
            statistic, p_value = chi_square_test(reference[col], window[col])
            rows.append(dict(window_id=window_id, start=start, end=start + window_size,
                             dataset=dataset, column=col, detector="chi2",
                             statistic=statistic, p_value=p_value,
                             alarm=int(p_value < threshold)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from drift.data import column_types, load_synthetic, reference_and_stream
    from drift.injection import inject_categorical, inject_sudden, shift_in_std

    syn = load_synthetic("adult")
    num, cat = column_types("adult", syn)
    ref, stream = reference_and_stream(syn, ref_size=2000)
    t0 = 4000
    n = len(num) + len(cat)
    print(f"{len(num)} numeric + {len(cat)} categorical columns "
          f"-> Bonferroni threshold p < {0.05 / n:.4f}\n")

    clean = window_tests(ref, stream, num, cat)
    print(f"No drift: {clean['alarm'].sum()} alarms in {len(clean)} tests (all false alarms)")
    if clean["alarm"].sum():
        print(clean[clean["alarm"] == 1][["window_id", "column", "p_value"]].to_string(index=False))

    # numeric drift, now tested alongside all 13 columns
    drifted = inject_sudden(stream, "age", t0, shift_in_std(ref, "age", 0.5))
    res = window_tests(ref, drifted, num, cat)
    first = res[(res["column"] == "age") & (res["alarm"] == 1)]["window_id"].min()
    print(f"\nSudden drift in age from record {t0:,}: first alarm in window {first} "
          f"(starts at record {first * 500:,})")

    # categorical drift
    col = "marital_status"
    drifted = inject_categorical(stream, col, t0, "Married-civ-spouse", "Never-married", frac=0.3)
    res = window_tests(ref, drifted, num, cat)
    mar = res[res["column"] == col][["window_id", "start", "statistic", "p_value", "alarm"]]
    print(f"\nCategorical drift in {col} from record {t0:,}:")
    print(mar.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    others = res[(res["column"] != col) & (res["alarm"] == 1)]
    print(f"\nAlarms in the other (unchanged) columns: {len(others)}")