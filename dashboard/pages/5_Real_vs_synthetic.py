"""DGIM page. Reads dgim, written from Matilda's dgim_service snapshots."""
import streamlit as st

import ui

ui.page_setup("Real vs synthetic stream (DGIM)")
st.caption("Share of records matching each tracked condition in the last N records, estimated with "
           "DGIM for the real and the synthetic stream. Lines that drift apart = the generator "
           "reproduces that condition badly.")
dataset, run = ui.filters()


@st.fragment(run_every=ui.REFRESH_SECONDS)
def body():
    df = ui.load("dgim", dataset, run)
    if ui.empty(df, "the DGIM service"):
        return
    preds = sorted(df["predicate"].unique())
    chosen = st.selectbox("Condition", preds)
    sub = df[df["predicate"] == chosen].copy()
    sub["series"] = sub["dataset"] + " · " + sub["source"]
    ui.line(sub, "records_seen", "rate", color="series", title=f"{chosen}: share in the sliding window")
    if "error_bound" in df and df["error_bound"].notna().any():
        st.caption(f"DGIM error bound: {df['error_bound'].dropna().iloc[-1]:.0%} of the true count.")
    latest = df.sort_values("time").groupby(["dataset", "predicate", "source"]).tail(1)
    table = latest.pivot_table(index=["dataset", "predicate"], columns="source", values="rate").reset_index()
    if {"real", "synthetic"} <= set(table.columns):
        table["gap"] = table["synthetic"] - table["real"]
    st.subheader("Latest window")
    st.dataframe(table, use_container_width=True, hide_index=True)


body()
