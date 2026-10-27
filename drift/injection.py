"""Inject known drift into a stream, so detectors can be scored against
ground truth (when the drift starts, and in which column).

All functions return a new DataFrame and leave the input unchanged.
Shift sizes for numeric columns are given in standard deviations of the
reference, so the same magnitude means the same thing for every column.
"""

import numpy as np
import pandas as pd


def shift_in_std(reference, col, n_std):
    """Convert a shift of `n_std` standard deviations into the column's units."""
    return n_std * reference[col].std()


def inject_sudden(stream, col, start, shift):
    """Sudden drift: add `shift` to `col` for every record from `start` on."""
    out = stream.copy()
    out[col] = out[col].astype(float)
    out.loc[start:, col] = out.loc[start:, col] + shift
    return out


def inject_gradual(stream, col, start, width, shift, seed=0):
    """Gradual drift: between `start` and `start + width`, each record is
    shifted with a probability that grows linearly from 0 to 1. After the
    transition every record is shifted."""
    rng = np.random.default_rng(seed)
    out = stream.copy()
    out[col] = out[col].astype(float)
    position = np.arange(len(out))
    prob = np.clip((position - start) / width, 0, 1)
    shifted = rng.random(len(out)) < prob
    out.loc[shifted, col] = out.loc[shifted, col] + shift
    return out


def inject_categorical(stream, col, start, from_cat, to_cat, frac, seed=0):
    """Categorical drift: from `start` on, relabel a fraction `frac` of the
    records whose value is `from_cat` as `to_cat`."""
    rng = np.random.default_rng(seed)
    out = stream.copy()
    candidates = out.index[(out.index >= start) & (out[col] == from_cat)]
    chosen = rng.choice(candidates, size=int(len(candidates) * frac), replace=False)
    out.loc[chosen, col] = to_cat
    return out


if __name__ == "__main__":
    from drift.data import load_synthetic, reference_and_stream

    ref, stream = reference_and_stream(load_synthetic("adult"), ref_size=2000)
    t0 = len(stream) // 2
    shift = shift_in_std(ref, "age", 0.5)
    print(f"drift starts at record t0 = {t0:,}; 0.5 std of age = {shift:.1f} years\n")

    def before_after(df, col):
        return df[col].iloc[:t0].mean(), df[col].iloc[t0:].mean()

    b, a = before_after(stream, "age")
    print(f"original      mean age before / after t0: {b:5.1f} / {a:5.1f}")
    b, a = before_after(inject_sudden(stream, "age", t0, shift), "age")
    print(f"sudden drift  mean age before / after t0: {b:5.1f} / {a:5.1f}")
    b, a = before_after(inject_gradual(stream, "age", t0, 2000, shift), "age")
    print(f"gradual drift mean age before / after t0: {b:5.1f} / {a:5.1f}  "
          "(smaller: the first 2,000 records are only partly shifted)")

    col, cat = "marital_status", "Married-civ-spouse"
    drifted = inject_categorical(stream, col, t0, cat, "Never-married", frac=0.3)
    share = lambda df, part: (df[col].iloc[part] == cat).mean()
    print(f"\nshare of {cat} before / after t0:")
    print(f"  original:           {share(stream, slice(None, t0)):.1%} / "
          f"{share(stream, slice(t0, None)):.1%}")
    print(f"  categorical drift:  {share(drifted, slice(None, t0)):.1%} / "
          f"{share(drifted, slice(t0, None)):.1%}")