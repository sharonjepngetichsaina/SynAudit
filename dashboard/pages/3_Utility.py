"""Utility page. Reads utility, written from Sharon's TSTR/TRTR results (Hoeffding tree)."""
import streamlit as st

import ui

ui.page_setup("Machine learning utility")
st.caption("A Hoeffding tree trained on synthetic data (TSTR) vs one trained on real data (TRTR), both "
           "tested on real data. Utility ratio = balanced accuracy synthetic / real; 1 = as good as real.")
dataset, run = ui.filters()


@st.fragment(run_every=ui.REFRESH_SECONDS)
def body():
    df = ui.load("utility", dataset, run)
    if ui.empty(df, "the utility service"):
        return
    latest = df.sort_values("window").groupby("dataset").tail(1)
    cols = st.columns(max(1, len(latest)))
    for col, (_, r) in zip(cols, latest.iterrows()):
        col.metric(f"{r['dataset']} · utility ratio", f"{r.get('utility_ratio', float('nan')):.2f}",
                   help=f"balanced accuracy {r.get('bal_acc_synthetic', float('nan')):.3f} synthetic vs "
                        f"{r.get('bal_acc_real', float('nan')):.3f} real")
    if "utility_ratio" in df:
        ui.line(df, "window", "utility_ratio", color="dataset", title="Utility ratio per window",
                ref=1.0, ref_label="real-data level")
    long = df.melt(id_vars=["window", "dataset"], value_vars=[c for c in ("bal_acc_synthetic", "bal_acc_real") if c in df],
                   var_name="trained_on", value_name="balanced_accuracy").dropna()
    long["trained_on"] = long["trained_on"].map({"bal_acc_synthetic": "synthetic", "bal_acc_real": "real"})
    long["series"] = long["dataset"] + " · " + long["trained_on"]
    ui.line(long, "window", "balanced_accuracy", color="series",
            title="Balanced accuracy on real test data", ref=0.5, ref_label="random guess", y_range=(0, 1))
    if "f1_synthetic" in df and (df["f1_synthetic"].fillna(0) == 0).all():
        st.warning("F1 of the synthetic model is 0 in every window: it predicts only one class.")
    with st.expander("Raw results"):
        st.dataframe(df, use_container_width=True, hide_index=True)


body()
