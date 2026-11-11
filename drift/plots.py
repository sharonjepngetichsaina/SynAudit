"""Figures for the report, made from the saved results in results/drift/.

  fig1_example_sudden_drift.png   one run: the signal, KS p-values and alarms
  fig2_calibration_tradeoff.png   false alarms vs. delay for each setting
  fig3_detection_rate.png         E3: detection rate vs. drift size
  fig4_detection_delay.png        E3: mean delay vs. drift size
  fig5_categorical.png            E3: categorical drift

Run from the repository root (after calibration and experiment):
    python -m drift.plots
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                    # write files only, no window
import matplotlib.pyplot as plt
import pandas as pd

from drift.data import column_types, load_synthetic, reference_and_stream
from drift.detectors import DEFAULTS, run_on_numeric
from drift.injection import inject_sudden, shift_in_std
from drift.window_tests import window_tests

RESULTS = Path("results/drift")
FIGURES = RESULTS / "figures"

# one colour per detector, the same in every figure
COLORS = {"ks": "#2a78d6", "chi2": "#2a78d6", "adwin": "#eb6834", "page_hinkley": "#1baf7a"}
LABELS = {"ks": "KS test (window)", "chi2": "Chi-square test (window)",
          "adwin": "ADWIN", "page_hinkley": "Page-Hinkley"}
MARKERS = {"ks": "o", "chi2": "o", "adwin": "s", "page_hinkley": "^"}

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
    "axes.edgecolor": "#8a8984", "font.size": 10, "figure.dpi": 150,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  {FIGURES / name}")


def fig_example():
    """One sudden drift in age: what the detectors see and when they fire."""
    syn = load_synthetic("adult")
    num, cat = column_types("adult", syn)
    ref, stream = reference_and_stream(syn, shuffle_seed=1)
    t0 = len(stream) // 2
    drifted = inject_sudden(stream, "age", t0, shift_in_std(ref, "age", 0.5))

    win = window_tests(ref, drifted, ["age"], n_tests=len(num) + len(cat))
    rec = run_on_numeric(ref, drifted, ["age"])
    alarms = {"ks": list(win.loc[win["alarm"] == 1, "end"] - 1),
              "adwin": rec[("adwin", "age")],
              "page_hinkley": rec[("page_hinkley", "age")]}
    threshold = 0.05 / (len(num) + len(cat))

    fig, axes = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1.2]})
    ax = axes[0]
    ax.plot(drifted["age"].rolling(500).mean(), color="#52514e", lw=1.5)
    ax.axhline(ref["age"].mean(), color="#8a8984", lw=1, ls=":")
    ax.set_ylabel("age\n(rolling mean, 500)")
    ax.set_title("Sudden drift in age (+0.5 std at t0) and when each detector fires", loc="left")

    ax = axes[1]
    ax.plot(win["end"], win["p_value"].clip(lower=1e-30), marker="o", ms=3.5, lw=1.5,
            color=COLORS["ks"])
    ax.axhline(threshold, color="#d03b3b", lw=1, ls="--")
    ax.text(0, threshold * 2.5, f"alarm threshold p < {threshold:.4f}", fontsize=8.5,
            color="#52514e")
    ax.set_yscale("log")
    ax.set_ylabel("KS p-value\n(log scale)")

    ax = axes[2]
    order = ["ks", "adwin", "page_hinkley"]
    for row, name in enumerate(order):
        ax.scatter(alarms[name], [row] * len(alarms[name]), marker="|", s=220,
                   color=COLORS[name], linewidths=2)
    ax.set_yticks(range(len(order)), [LABELS[n] for n in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("record in stream")

    for ax in axes:
        ax.axvline(t0, color="#8a8984", lw=1.2, ls="--")
    axes[0].annotate("drift injected (t0)", (t0, 1), xycoords=("data", "axes fraction"),
                     xytext=(8, -14), textcoords="offset points", fontsize=8.5, color="#52514e")
    save(fig, "fig1_example_sudden_drift.png")


def fig_calibration():
    """Each parameter setting as one point: false alarms vs. detection delay.
    Only the settings we chose (the defaults in drift/detectors.py) are labelled."""
    grid = pd.read_csv(RESULTS / "calibration_grid.csv")
    summary = (grid.groupby(["detector", "delta", "threshold"], dropna=False)
               .agg(false_alarms=("false_alarms_clean", "mean"), delay=("delay", "mean"))
               .reset_index())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, part in summary.groupby("detector"):
        ax.scatter(part["delay"], part["false_alarms"], s=55, color=COLORS[name],
                   marker=MARKERS[name], label=LABELS[name], edgecolor="white", linewidth=1)
        chosen = DEFAULTS[name]
        pick = part[part["delta"] == chosen["delta"]]
        if "threshold" in chosen:
            pick = pick[pick["threshold"] == chosen["threshold"]]
        for _, r in pick.iterrows():
            text = f"chosen: δ={r['delta']:g}" + (
                "" if pd.isna(r["threshold"]) else f", λ={r['threshold']:g}")
            ax.annotate(text, (r["delay"], r["false_alarms"]), xytext=(10, 22),
                        textcoords="offset points", fontsize=8.5, color="#0b0b0b",
                        arrowprops=dict(arrowstyle="-", color="#8a8984", lw=0.8))
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("mean detection delay for a +0.5 std shift (records)")
    ax.set_ylabel("false alarms on drift-free data\n(per run, 5 numeric columns)")
    ax.set_title("Calibration: faster detection costs false alarms", loc="left")
    ax.legend(frameon=False)
    save(fig, "fig2_calibration_tradeoff.png")


def per_magnitude(numeric, value):
    return (numeric.groupby(["drift", "magnitude_std", "detector"])[value].mean()
            .reset_index())


def fig_numeric(numeric, value, ylabel, title, name):
    data = per_magnitude(numeric, value)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, kind in zip(axes, ["sudden", "gradual"]):
        for det in ["ks", "adwin", "page_hinkley"]:
            part = data[(data["drift"] == kind) & (data["detector"] == det)]
            ax.plot(part["magnitude_std"], part[value], marker=MARKERS[det], ms=6, lw=2,
                    color=COLORS[det], label=LABELS[det])
        ax.set_title(f"{kind.capitalize()} drift", loc="left")
        ax.set_xlabel("drift size (standard deviations)")
        ax.set_xticks(sorted(numeric["magnitude_std"].unique()))
    axes[0].set_ylabel(ylabel)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    save(fig, name)


def fig_categorical(categorical):
    data = (categorical.groupby(["fraction", "detector"])
            .agg(rate=("detected", "mean"), delay=("delay", "mean")).reset_index())
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for det in ["chi2", "adwin"]:
        part = data[data["detector"] == det]
        x = part["fraction"] * 100
        axes[0].plot(x, part["rate"], marker=MARKERS[det], ms=6, lw=2,
                     color=COLORS[det], label=LABELS[det])
        axes[1].plot(x, part["delay"], marker=MARKERS[det], ms=6, lw=2,
                     color=COLORS[det], label=LABELS[det])
    axes[0].set_ylabel("detection rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("mean delay (records, detected runs)")
    for ax in axes:
        ax.set_xlabel("records relabelled (%)")
        ax.set_xticks(sorted(data["fraction"].unique() * 100))
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("Categorical drift in marital_status", x=0.01, ha="left", fontsize=11)
    save(fig, "fig5_categorical.png")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("Writing figures:")
    fig_example()
    fig_calibration()
    numeric = pd.read_csv(RESULTS / "e3_numeric.csv")
    fig_numeric(numeric, "detected", "detection rate",
                "Detection rate: share of runs where the drift was found within 2,000 records",
                "fig3_detection_rate.png")
    fig_numeric(numeric, "delay", "mean delay (records, detected runs)",
                "Detection delay: bigger shifts are found faster",
                "fig4_detection_delay.png")
    fig_categorical(pd.read_csv(RESULTS / "e3_categorical.csv"))


if __name__ == "__main__":
    main()