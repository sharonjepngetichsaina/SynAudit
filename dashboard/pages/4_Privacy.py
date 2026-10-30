"""Privacy page. Reads privacy, written from Sharon's copy check (exact and Bloom filter)."""
import streamlit as st

import ui

ui.page_setup("Privacy risk")
st.caption("Share of synthetic records that copy a real training record, counted exactly and with a "
           "Bloom filter (fast, may over-count through false positives).")
dataset, run = ui.filters()


@st.fragment(run_every=ui.REFRESH_SECONDS)
def body():
    df = ui.load("privacy", dataset, run)
    if ui.empty(df, "the privacy service"):
        return
    latest = df.sort_values("window").groupby("dataset").tail(1)
    cols = st.columns(max(1, len(latest)))
    for col, (_, r) in zip(cols, latest.iterrows()):
        col.metric(f"{r['dataset']} · exact copy rate", f"{r.get('exact_copy_rate', float('nan')):.2%}",
                   help=f"Bloom filter estimate: {r.get('bloom_copy_rate', float('nan')):.2%}")
    if (latest.get("exact_copy_rate", 0) > 0.01).any():
        st.warning("More than 1% of synthetic records are copies of real records.")
    long = df.melt(id_vars=["window", "dataset"], value_vars=[c for c in ("exact_copy_rate", "bloom_copy_rate") if c in df],
                   var_name="method", value_name="copy_rate").dropna()
    long["method"] = long["method"].map({"exact_copy_rate": "exact", "bloom_copy_rate": "Bloom filter"})
    long["series"] = long["dataset"] + " · " + long["method"]
    ui.line(long, "window", "copy_rate", color="series", title="Copy rate per window: exact vs Bloom filter")
    with st.expander("Raw results"):
        st.dataframe(df, use_container_width=True, hide_index=True)


body()
