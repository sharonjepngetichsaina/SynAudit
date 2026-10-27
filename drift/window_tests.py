"""Window-level drift tests: compare each window of the stream with the
reference sample using a two-sample statistical test per column.

Numeric columns: two-sample Kolmogorov-Smirnov (KS) test.

Each window tests several columns, so the significance level is
Bonferroni-corrected: a column raises an alarm when p < alpha / n_tests.
"""

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


def window_tests(reference, stream, num_cols, window_size=500, alpha=0.05,
                 n_tests=None, dataset="adult"):
    """Run the KS test for every numeric column in every window of the stream.

    Returns one row per (window, column), shaped like the `drift_events`
    records we will write to InfluxDB. `n_tests` is the number of tests per
    window used for the Bonferroni correction (default: number of columns).
    """
    n_tests = n_tests or len(num_cols)
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
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from drift.data import column_types, load_synthetic, reference_and_stream
    from drift.injection import inject_sudden, shift_in_std

    syn = load_synthetic("adult")
    num, _ = column_types("adult", syn)
    ref, stream = reference_and_stream(syn, ref_size=2000)
    t0 = 4000
    print(f"{len(num)} numeric columns -> Bonferroni threshold p < {0.05 / len(num):.3f}\n")

    clean = window_tests(ref, stream, num)
    print(f"No drift: {clean['alarm'].sum()} alarms in {len(clean)} tests (all false alarms)")

    drifted = inject_sudden(stream, "age", t0, shift_in_std(ref, "age", 0.5))
    res = window_tests(ref, drifted, num)
    age = res[res["column"] == "age"][["window_id", "start", "statistic", "p_value", "alarm"]]
    print(f"\nSudden drift in age (+0.5 std) from record {t0:,}:")
    print(age.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    others = res[(res["column"] != "age") & (res["alarm"] == 1)]
    print(f"\nAlarms in the other (unchanged) columns: {len(others)}")