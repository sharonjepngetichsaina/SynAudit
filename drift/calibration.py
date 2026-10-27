"""Calibrate the drift detectors on drift-free data.

1. False alarm rate of every detector on the synthetic stream with no
   injected drift (every alarm is a false alarm), over several seeds.
2. Parameter grids for Page-Hinkley (delta, threshold) and ADWIN (delta):
   false alarms on drift-free data versus detection delay for a reference
   drift (sudden +0.5 std shift in age). This shows the trade-off and
   justifies the defaults in drift/detectors.py.

Run from the repository root:  python -m drift.calibration
Results are written to results/drift/.
"""

from pathlib import Path

import pandas as pd

from drift.data import column_types, load_synthetic, reference_and_stream
from drift.detectors import run_on_numeric
from drift.evaluation import evaluate_alarms
from drift.injection import inject_sudden, shift_in_std
from drift.window_tests import window_tests

OUT = Path("results/drift")
SEEDS = [1, 2, 3, 4, 5]
REF_SIZE = 2000
T0 = 4000                       # change point of the reference drift


def false_alarm_rates(syn, num, cat):
    """False alarms of every detector on drift-free streams."""
    rows = []
    for seed in SEEDS:
        ref, stream = reference_and_stream(syn, REF_SIZE, shuffle_seed=seed)
        win = window_tests(ref, stream, num, cat)
        for col, group in win.groupby("column"):
            rows.append(dict(seed=seed, detector=group["detector"].iloc[0],
                             column=col, false_alarms=int(group["alarm"].sum()),
                             records=len(stream)))
        for (name, col), alarms in run_on_numeric(ref, stream, num).items():
            rows.append(dict(seed=seed, detector=name, column=col,
                             false_alarms=len(alarms), records=len(stream)))
    return pd.DataFrame(rows)


def parameter_grid(syn, num, name, grid):
    """For each setting: false alarms on clean data (all numeric columns)
    and detection delay for the reference drift in age."""
    rows = []
    for params in grid:
        for seed in SEEDS:
            ref, stream = reference_and_stream(syn, REF_SIZE, shuffle_seed=seed)
            setting = {name: params}
            clean = run_on_numeric(ref, stream, num, (name,), setting)
            drifted = inject_sudden(stream, "age", T0, shift_in_std(ref, "age", 0.5))
            alarms = run_on_numeric(ref, drifted, ["age"], (name,), setting)[(name, "age")]
            score = evaluate_alarms(alarms, T0)
            rows.append(dict(detector=name, **params, seed=seed,
                             false_alarms_clean=sum(len(a) for a in clean.values()),
                             delay=score["delay"], detected=score["detected"]))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    syn = load_synthetic("adult")
    num, cat = column_types("adult", syn)

    # 1. false alarm rates with the default settings
    fa = false_alarm_rates(syn, num, cat)
    fa.to_csv(OUT / "calibration_false_alarms.csv", index=False)
    per_detector = (fa.groupby("detector")
                    .agg(columns=("column", "nunique"),
                         false_alarms=("false_alarms", "sum"),
                         records=("records", "sum"))
                    .assign(per_10k_records_per_column=lambda d:
                            (d["false_alarms"] / d["records"] * 10_000).round(2)))
    print(f"False alarms on drift-free data ({len(SEEDS)} seeds, defaults):")
    print(per_detector[["columns", "false_alarms", "per_10k_records_per_column"]].to_string())
    worst = fa[fa["false_alarms"] > 0].groupby(["detector", "column"])["false_alarms"].sum()
    if len(worst):
        print("\nColumns with false alarms:")
        print(worst.to_string())

    # 2. parameter grids
    ph_grid = [dict(delta=d, threshold=t) for d in (0.05, 0.1, 0.25)
               for t in (25, 50, 100, 200)]
    adwin_grid = [dict(delta=d) for d in (0.0002, 0.002, 0.02, 0.2)]
    grid = pd.concat([parameter_grid(syn, num, "page_hinkley", ph_grid),
                      parameter_grid(syn, num, "adwin", adwin_grid)])
    grid.to_csv(OUT / "calibration_grid.csv", index=False)

    keys = ["detector", "delta", "threshold"]
    summary = (grid.astype({"threshold": object}).fillna({"threshold": "-"})
               .groupby(keys, sort=False)
               .agg(false_alarms_clean=("false_alarms_clean", "mean"),
                    detection_rate=("detected", "mean"),
                    mean_delay=("delay", "mean"))
               .round(1))
    print(f"\nParameter grid ({len(SEEDS)} seeds; false alarms summed over "
          f"{len(num)} numeric columns; delay for +0.5 std in age):")
    print(summary.to_string())
    print(f"\nSaved: {OUT / 'calibration_false_alarms.csv'}, {OUT / 'calibration_grid.csv'}")


if __name__ == "__main__":
    main()