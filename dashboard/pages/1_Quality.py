"""Quality page. Reads what Diana's quality service writes (docs/INFLUX_SCHEMA.md)."""
import plotly.express as px
import streamlit as st

import ui

ui.page_setup("Statistical quality")
st.caption("How closely each synthetic window reproduces the real data, from the quality service "
           "(SMD / TVD distances, SDMetrics, CluStream and DenStream). Dashed lines = alert thresholds.")
dataset, run = ui.filters()

# field -> (label, threshold, alert when 'above' or 'below')
WATCH = {
    "sdmetrics_overall": ("SDMetrics overall", 0.8, "below"),
    "smd_max": ("Worst SMD (numeric)", 0.5, "above"),
    "tvd_max": ("Worst TVD (categorical)", 0.1, "above"),
    "corr_diff_mean": ("Correlation difference", 0.1, "above"),
    "zero_gap_max": ("Worst zero-share gap", 0.1, "above"),
    "consistency_min": ("Consistency (adult)", 0.99, "below"),
    "clustream_coverage_ratio": ("CluStream coverage ratio", 1.5, "above"),
    "denstream_coverage_ratio": ("DenStream coverage ratio", 1.5, "above"),
}


def last(df, field):
    s = df[field].dropna() if field in df else None
    return None if s is None or s.empty else s


@st.fragment(run_every=ui.REFRESH_SECONDS)
def body():
    df = ui.load("synaudit_quality", dataset, None)
    if ui.empty(df, "the quality service"):
        return

    cards = [f for f in ("sdmetrics_overall", "smd_max", "tvd_max", "consistency_min") if last(df, f) is not None]
    for col, field in zip(st.columns(max(1, len(cards))), cards):
        s = last(df, field)
        col.metric(WATCH[field][0], f"{s.iloc[-1]:.3f}", ui.delta(s),
                   delta_color="normal" if WATCH[field][2] == "below" else "inverse")

    latest = df.sort_values("window_id").groupby("dataset").tail(1)
    alerts = []
    for _, row in latest.iterrows():
        for field, (label, thr, side) in WATCH.items():
            v = row.get(field)
            if v is not None and v == v and ((side == "above" and v > thr) or (side == "below" and v < thr)):
                alerts.append(f"{row['dataset']}: {label} = {v:.3f} (threshold {thr})")
    if alerts:
        st.warning("Latest window outside the threshold:\n\n- " + "\n- ".join(alerts))

    st.subheader("Distribution distance per window")
    left, right = st.columns(2)
    for i, field in enumerate(("smd_max", "tvd_max", "corr_diff_mean", "sdmetrics_overall")):
        if field in df and df[field].notna().any():
            with (left if i % 2 == 0 else right):
                label, thr, _ = WATCH[field]
                ui.line(df.dropna(subset=[field]), "window_id", field, color="dataset",
                        title=label, ref=thr, ref_label="threshold")

    clus = [f for f in ("clustream_coverage_ratio", "denstream_coverage_ratio") if f in df]
    if clus:
        st.subheader("Cluster structure: CluStream vs DenStream")
        long = df.melt(id_vars=["window_id", "dataset"], value_vars=clus,
                       var_name="algorithm", value_name="coverage_ratio").dropna()
        long["algorithm"] = long["algorithm"].str.replace("_coverage_ratio", "", regex=False)
        long["series"] = long["dataset"] + " · " + long["algorithm"]
        ui.line(long, "window_id", "coverage_ratio", color="series",
                title="Coverage ratio (about 1 = synthetic data covers the real clusters)",
                ref=1.5, ref_label="threshold")

    cols = ui.load("synaudit_column_quality", dataset, None)
    if not cols.empty:
        st.subheader("Per column, latest window")
        value_cols = [c for c in ("smd", "tvd", "zero_gap") if c in cols]
        tidy = cols.melt(id_vars=["time", "dataset", "column"], value_vars=value_cols,
                         var_name="metric", value_name="value").dropna()
        tidy = tidy.sort_values("time").groupby(["dataset", "column", "metric"]).tail(1)
        fig = px.bar(tidy, x="value", y="column", color="dataset", facet_col="metric",
                     orientation="h", barmode="group", title="Distance per column (0 = identical)")
        fig.update_xaxes(matches=None)
        fig.update_layout(height=120 + 26 * tidy["column"].nunique(), margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw results"):
        st.dataframe(df, use_container_width=True, hide_index=True)


body()
