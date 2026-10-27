"""Record-level change detectors from the River library.

Unlike the window tests, these read the stream one value at a time and keep
a memory of what they have seen, so evidence of a change accumulates.

ADWIN (Bifet & Gavalda, 2007)
    Keeps a window of recent values whose size adapts. Whenever two parts
    of the window (older vs. newer) have means that differ by more than a
    bound set by the confidence parameter delta, the older part is dropped
    and a change is signalled. Small delta = fewer false alarms.

Page-Hinkley (Page, 1954)
    Sums up how far each value lies from the running mean, minus a
    tolerance delta. When that cumulative sum rises more than a threshold
    lambda above its lowest point, a change is signalled. Larger lambda =
    fewer false alarms but a longer delay.

Values are standardised with the reference mean and standard deviation,
so the same parameters work for every column (a shift of 1 = 1 std).
"""

import numpy as np
from river import drift

# Page-Hinkley defaults: delta ~ half the smallest shift (in std) we want to
# catch; threshold chosen on drift-free data to keep false alarms rare.
DEFAULTS = {
    "adwin": {"delta": 0.002},
    "page_hinkley": {"min_instances": 30, "delta": 0.1, "threshold": 100.0},
}


def make_detector(name, **params):
    """Create a fresh detector; `params` override the defaults."""
    settings = DEFAULTS[name] | params
    if name == "adwin":
        return drift.ADWIN(**settings)
    if name == "page_hinkley":
        return drift.PageHinkley(**settings)
    raise ValueError(f"unknown detector: {name}")


def run_detector(values, name, ref_mean, ref_std, **params):
    """Feed values one by one into a detector; return the alarm positions.

    Missing values are skipped (they do not move the detector), but positions
    still refer to the original stream, so delays can be measured.
    """
    detector = make_detector(name, **params)
    scale = ref_std if ref_std > 0 else 1.0
    alarms = []
    for position, x in enumerate(values):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            continue
        detector.update((x - ref_mean) / scale)
        if detector.drift_detected:
            alarms.append(position)
    return alarms


def run_on_numeric(reference, stream, columns, detectors=("adwin", "page_hinkley"),
                   params=None):
    """Run every detector on every numeric column.

    Returns {(detector, column): [alarm positions]}.
    """
    params = params or {}
    alarms = {}
    for col in columns:
        mean, std = reference[col].mean(), reference[col].std()
        values = stream[col].astype(float).to_numpy()
        for name in detectors:
            alarms[(name, col)] = run_detector(values, name, mean, std,
                                               **params.get(name, {}))
    return alarms


def run_on_category(reference, stream, column, category, name="adwin", **params):
    """Watch the share of one category.

    The stream becomes an indicator: 1 if the record has `category`, else 0.
    The mean of that indicator is the category's share, so the detector
    signals when the share changes.
    """
    ref_ind = (reference[column] == category).astype(float)
    values = (stream[column] == category).astype(float).to_numpy()
    return run_detector(values, name, ref_ind.mean(), ref_ind.std(), **params)


if __name__ == "__main__":
    from drift.data import column_types, load_synthetic, reference_and_stream
    from drift.injection import inject_categorical, inject_sudden, shift_in_std

    syn = load_synthetic("adult")
    num, _ = column_types("adult", syn)
    ref, stream = reference_and_stream(syn, ref_size=2000)
    t0 = 4000

    print("No drift (every alarm is a false alarm):")
    clean = run_on_numeric(ref, stream, num)
    for (name, col), alarms in clean.items():
        print(f"  {name:<13} {col:<15} {len(alarms)} alarms {alarms[:5]}")

    print(f"\nSudden drift in age (+0.5 std) from record {t0:,}:")
    drifted = inject_sudden(stream, "age", t0, shift_in_std(ref, "age", 0.5))
    for (name, col), alarms in run_on_numeric(ref, drifted, ["age"]).items():
        after = [a for a in alarms if a >= t0]
        delay = after[0] - t0 if after else None
        print(f"  {name:<13} alarms at {alarms[:5]} -> delay {delay} records")

    col, cat = "marital_status", "Married-civ-spouse"
    print(f"\nCategorical drift in {col} from record {t0:,} (ADWIN on the share of {cat}):")
    drifted = inject_categorical(stream, col, t0, cat, "Never-married", frac=0.3)
    clean_alarms = run_on_category(ref, stream, col, cat)
    alarms = run_on_category(ref, drifted, col, cat)
    after = [a for a in alarms if a >= t0]
    print(f"  no drift: {len(clean_alarms)} alarms")
    print(f"  drift:    alarms at {alarms[:5]} -> delay {after[0] - t0 if after else None} records")