"""SynAudit dashboard, overview page.  Run: streamlit run dashboard/app.py"""
import streamlit as st

import ui

ui.page_setup("SynAudit overview")
st.caption("Live evaluation of synthetic data: quality, drift, ML utility and privacy. "
           "Open a page in the sidebar for details.")
dataset, run = ui.filters()


@st.fragment(run_every=ui.REFRESH_SECONDS)
def overview():
    q, d = ui.load("synaudit_quality", dataset, run), ui.load("drift_events", dataset, run)
    u, p = ui.load("utility", dataset, run), ui.load("privacy", dataset, run)
    if q.empty and d.empty and u.empty and p.empty:
        st.info("InfluxDB is empty. Start the pipeline, or load demo data with "
                "`python scripts/seed_dashboard_demo.py`.")
        return
    overall = q.dropna(subset=["sdmetrics_overall"]) if "sdmetrics_overall" in q else q.iloc[0:0]
    ratio = u.sort_values("window").groupby("window")["utility_ratio"].mean() if "utility_ratio" in u else None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quality (SDMetrics overall)", f"{overall['sdmetrics_overall'].iloc[-1]:.2f}" if not overall.empty else "–",
              ui.delta(overall["sdmetrics_overall"]) if not overall.empty else None)
    c2.metric("Drift alarms", int(d["alarm"].fillna(0).astype(int).sum()) if not d.empty else "–")
    c3.metric("Utility ratio", f"{ratio.iloc[-1]:.2f}" if ratio is not None else "–",
              ui.delta(ratio) if ratio is not None else None)
    has_copies = "exact_copy_rate" in p and p["exact_copy_rate"].notna().any()
    c4.metric("Exact copy rate", f"{p['exact_copy_rate'].dropna().iloc[-1]:.2%}" if has_copies else "–",
              ui.delta(p["exact_copy_rate"]) if has_copies else None, delta_color="inverse")
    st.divider()
    left, right = st.columns(2)
    with left:
        if not overall.empty:
            ui.line(overall, "window_id", "sdmetrics_overall", color="dataset",
                    title="Quality per window (SDMetrics overall)", ref=0.8, ref_label="threshold", y_range=(0, 1))
        if "utility_ratio" in u:
            ui.line(u, "window", "utility_ratio", color="dataset", title="Utility ratio per window",
                    ref=1.0, ref_label="real-data level")
    with right:
        if not d.empty:
            per = (d.assign(alarm=d["alarm"].fillna(0).astype(int))
                   .groupby(["dataset", "window_id"])["alarm"].sum().reset_index(name="alarms"))
            ui.line(per, "window_id", "alarms", color="dataset", title="Drift alarms per window")
        if has_copies:
            ui.line(p, "window", "exact_copy_rate", color="dataset", title="Exact copy rate per window")


overview()
