import streamlit as st

import ui

ui.page_setup("Statistical quality")
st.caption("How closely each synthetic batch reproduces the real data (SDMetrics, 1.0 = identical).")
dataset, run = ui.filters()


@st.fragment(run_every=ui.REFRESH_SECONDS)
def body():
    df = ui.load("quality", dataset, run)
    if ui.empty(df, "the quality service"):
        return
    latest = df[df["batch_id"] == df["batch_id"].max()]
    cols = st.columns(max(1, latest["metric"].nunique()))
    for col, (metric, grp) in zip(cols, latest.groupby("metric")):
        col.metric(metric.replace("_", " ").capitalize(), f"{grp['score'].mean():.3f}",
                   ui.delta(df[df["metric"] == metric]["score"]))
    ui.line(df, "batch_id", "score", color="metric", title="Score per batch", y_range=(0, 1))
    with st.expander("Raw results"):
        st.dataframe(df, use_container_width=True, hide_index=True)


body()
