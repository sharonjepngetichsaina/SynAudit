"""Score drift detectors against a known change point t0 (ground truth).

Metrics per run:
  detection delay : first alarm at or after t0, minus t0 (in records)
  false alarms    : alarms before t0
  detected        : an alarm within `max_delay` records after t0
Over repeated runs, the share of detected runs is the detection rate.
"""

import numpy as np
import pandas as pd

from drift.detectors import run_on_numeric
from drift.window_tests import window_tests


def evaluate_alarms(alarms, t0, max_delay=2000):
    """Score alarm positions against the change point t0.

    If t0 is None (no drift injected), every alarm is a false alarm.
    """
    alarms = np.asarray(alarms, dtype=int)
    if t0 is None:
        return dict(false_alarms=len(alarms), delay=np.nan, detected=False)
    before = alarms[alarms < t0]
    after = alarms[(alarms >= t0) & (alarms <= t0 + max_delay)]
    detected = len(after) > 0
    return dict(false_alarms=len(before),
                delay=float(after[0] - t0) if detected else np.nan,
                detected=detected)


def window_alarm_positions(results, column):
    """Alarm positions of a window test: the last record of each alarming
    window, because that is when the test can first raise the alarm."""
    hits = results[(results["column"] == column) & (results["alarm"] == 1)]
    return (hits["end"] - 1).tolist()


def score_numeric_run(reference, stream, column, t0, n_tests, max_delay=2000):
    """Run the KS window test, ADWIN and Page-Hinkley on one column and
    score all three. Returns one dict per detector."""
    win = window_tests(reference, stream, [column], n_tests=n_tests)
    alarms = {"ks": window_alarm_positions(win, column)}
    for (name, _), positions in run_on_numeric(reference, stream, [column]).items():
        alarms[name] = positions
    return [dict(detector=name, **evaluate_alarms(a, t0, max_delay))
            for name, a in alarms.items()]


def summarise(results):
    """Average the runs per detector (and any other grouping columns)."""
    keys = [c for c in results.columns
            if c not in ("seed", "false_alarms", "delay", "detected")]
    return (results.groupby(keys)
            .agg(runs=("detected", "size"),
                 detection_rate=("detected", "mean"),
                 mean_delay=("delay", "mean"),
                 false_alarms_per_run=("false_alarms", "mean"))
            .round(2))


if __name__ == "__main__":
    from drift.data import column_types, load_synthetic, reference_and_stream
    from drift.injection import inject_sudden, shift_in_std

    syn = load_synthetic("adult")
    num, cat = column_types("adult", syn)
    n_tests = len(num) + len(cat)          # same Bonferroni threshold as the full system
    t0, seeds = 4000, [1, 2, 3, 4, 5]

    rows = []
    for seed in seeds:
        ref, stream = reference_and_stream(syn, shuffle_seed=seed)
        drifted = inject_sudden(stream, "age", t0, shift_in_std(ref, "age", 0.5))
        for score in score_numeric_run(ref, drifted, "age", t0, n_tests):
            rows.append(dict(seed=seed, **score))

    results = pd.DataFrame(rows)
    print(f"Sudden drift in age (+0.5 std) at t0 = {t0:,}, {len(seeds)} seeds\n")
    print("Per run:")
    print(results.to_string(index=False))
    print("\nSummary:")
    print(summarise(results).to_string())