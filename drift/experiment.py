"""Experiment E3: how well do the detectors find injected drift?

Numeric drift (columns age and hours_per_week)
    type:       sudden, or gradual (new values take over over 2,000 records)
    magnitude:  0.1, 0.25, 0.5, 1.0 standard deviations
    detectors:  KS window test, ADWIN, Page-Hinkley

Categorical drift (column marital_status)
    a fraction (10, 20, 30 %) of 'Married-civ-spouse' records relabelled
    as 'Never-married'
    detectors:  chi-square window test, ADWIN on the category's share

Every setting is repeated with several seeds. Each seed shuffles the data
and places the change point t0 at a random record, so t0 does not line up
with the window boundaries (which would favour the window tests).

Run from the repository root:  python -m drift.experiment
Results are written to results/drift/.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from drift.data import column_types, load_synthetic, reference_and_stream
from drift.detectors import run_on_category
from drift.evaluation import (evaluate_alarms, score_numeric_run, summarise,
                              window_alarm_positions)
from drift.injection import (inject_categorical, inject_gradual, inject_sudden,
                             shift_in_std)
from drift.window_tests import window_tests

OUT = Path("results/drift")
SEEDS = list(range(1, 11))
MAX_DELAY = 2000
GRADUAL_WIDTH = 2000

NUMERIC_COLUMNS = ["age", "hours_per_week"]
MAGNITUDES = [0.1, 0.25, 0.5, 1.0]

CAT_COLUMN, FROM_CAT, TO_CAT = "marital_status", "Married-civ-spouse", "Never-married"
FRACTIONS = [0.1, 0.2, 0.3]


def random_t0(stream_length, seed):
    """A change point with room before it (for false alarms) and after it
    (for a detection within MAX_DELAY, plus the gradual transition)."""
    low = stream_length // 5
    high = stream_length - MAX_DELAY - GRADUAL_WIDTH
    if high <= low:
        raise ValueError(f"stream of {stream_length:,} records is too short for this experiment")
    rng = np.random.default_rng(1000 + seed)
    return int(rng.integers(low, high))


def numeric_experiment(syn, n_tests):
    rows = []
    for seed in SEEDS:
        ref, stream = reference_and_stream(syn, shuffle_seed=seed)
        t0 = random_t0(len(stream), seed)
        for col in NUMERIC_COLUMNS:
            for kind in ("sudden", "gradual"):
                for mag in MAGNITUDES:
                    shift = shift_in_std(ref, col, mag)
                    if kind == "sudden":
                        drifted = inject_sudden(stream, col, t0, shift)
                    else:
                        drifted = inject_gradual(stream, col, t0, GRADUAL_WIDTH, shift, seed=seed)
                    for score in score_numeric_run(ref, drifted, col, t0, n_tests, MAX_DELAY):
                        rows.append(dict(seed=seed, t0=t0, column=col, drift=kind,
                                         magnitude_std=mag, **score))
    return pd.DataFrame(rows)


def categorical_experiment(syn, n_tests):
    rows = []
    for seed in SEEDS:
        ref, stream = reference_and_stream(syn, shuffle_seed=seed)
        t0 = random_t0(len(stream), seed)
        for frac in FRACTIONS:
            drifted = inject_categorical(stream, CAT_COLUMN, t0, FROM_CAT, TO_CAT, frac, seed=seed)
            win = window_tests(ref, drifted, [], [CAT_COLUMN], n_tests=n_tests)
            alarms = {"chi2": window_alarm_positions(win, CAT_COLUMN),
                      "adwin": run_on_category(ref, drifted, CAT_COLUMN, FROM_CAT)}
            for name, positions in alarms.items():
                rows.append(dict(seed=seed, t0=t0, column=CAT_COLUMN, fraction=frac,
                                 detector=name, **evaluate_alarms(positions, t0, MAX_DELAY)))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    syn = load_synthetic("adult")
    num, cat = column_types("adult", syn)
    n_tests = len(num) + len(cat)          # same Bonferroni threshold as the full system
    print(f"{len(syn):,} synthetic records, {len(SEEDS)} seeds, "
          f"detection counts only within {MAX_DELAY:,} records after t0\n")

    numeric = numeric_experiment(syn, n_tests)
    numeric.to_csv(OUT / "e3_numeric.csv", index=False)
    print("Numeric drift (both columns together):")
    print(summarise(numeric.drop(columns=["t0", "column"])).to_string())

    categorical = categorical_experiment(syn, n_tests)
    categorical.to_csv(OUT / "e3_categorical.csv", index=False)
    print(f"\nCategorical drift in {CAT_COLUMN} ({FROM_CAT} -> {TO_CAT}):")
    print(summarise(categorical.drop(columns=["t0", "column"])).to_string())

    print(f"\nSaved: {OUT / 'e3_numeric.csv'}, {OUT / 'e3_categorical.csv'}")


if __name__ == "__main__":
    main()