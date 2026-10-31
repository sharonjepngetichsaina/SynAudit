"""Sketches page. Reads sketches and sketch_freq, written by Alina's sketch service."""
import plotly.express as px
import streamlit as st

import ui

ui.page_setup("Stream sketches")
st.caption("Count-Min Sketch estimates how often each category appears; HyperLogLog estimates how many "
           "distinct values a column has. Both run in fixed memory on the real and the synthetic stream.")
dataset, run = ui.filters()


@st.fragment(run_every=ui.REFRESH_SECONDS)
def body():
    sk = ui.load("sketches", dataset, run)
    fq = ui.load("sketch_freq", dataset, run)
    if ui.empty(sk, "the sketch service"):
        return
    datasets = sorted(sk["dataset"].unique())
    ds = datasets[0] if len(datasets) == 1 else st.selectbox("Dataset", datasets)
    sk, fq = sk[sk["dataset"] == ds], fq[fq["dataset"] == ds] if not fq.empty else fq
    col = st.selectbox("Column", sorted(sk["column"].unique()))

    latest = sk[sk["column"] == col].sort_values("time").groupby("source").tail(1).set_index("source")
    c1, c2, c3 = st.columns(3)
    for c, src in ((c1, "real"), (c2, "synthetic")):
        if src in latest.index:
            r = latest.loc[src]
            c.metric(f"Distinct values · {src}", f"{r['hll_estimate']:.0f}",
                     help=f"exact {r.get('hll_exact', '–')}, HyperLogLog error {r.get('hll_rel_error', float('nan')):.1%}")
    if {"real", "synthetic"} <= set(latest.index):
        gap = latest.loc["synthetic", "hll_estimate"] - latest.loc["real", "hll_estimate"]
        c3.metric("Synthetic minus real", f"{gap:+.0f}", help="negative = synthetic data has less variety")

    if not fq.empty:
        f = fq[fq["column"] == col].sort_values("time").groupby("category").tail(1)
        f = f.assign(gap=f["synthetic_share"] - f["real_share"]).sort_values("real_share", ascending=False).head(20)
        long = f.melt(id_vars=["category"], value_vars=["real_share", "synthetic_share"],
                      var_name="stream", value_name="share")
        long["stream"] = long["stream"].str.replace("_share", "", regex=False)
        fig = px.bar(long, x="share", y="category", color="stream", orientation="h", barmode="group",
                     title=f"{col}: share of each category (Count-Min Sketch estimate)")
        fig.update_layout(height=140 + 26 * len(f), yaxis={"categoryorder": "total ascending"},
                          margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("How accurate are the sketches?")
    acc = sk[sk["column"] == col].copy()
    left, right = st.columns(2)
    with left:
        if "hll_rel_error" in acc:
            ui.line(acc.dropna(subset=["hll_rel_error"]), "records_seen", "hll_rel_error", color="source",
                    title="HyperLogLog relative error vs exact count")
    with right:
        if "cms_max_error_share" in acc:
            eps = acc["cms_epsilon"].dropna().iloc[-1] if "cms_epsilon" in acc else None
            ui.line(acc.dropna(subset=["cms_max_error_share"]), "records_seen", "cms_max_error_share",
                    color="source", title="Count-Min worst over-count (share of stream)",
                    ref=eps, ref_label="guarantee epsilon")
    r = latest.iloc[-1]
    if "exact_memory_bytes" in r and r["exact_memory_bytes"]:
        st.caption(f"Memory for this column: Count-Min {int(r['cms_memory_bytes']):,} B, HyperLogLog "
                   f"{int(r['hll_memory_bytes']):,} B, exact counting {int(r['exact_memory_bytes']):,} B.")


body()
