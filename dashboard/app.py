"""SynAudit dashboard, overview page.  Run: streamlit run dashboard/app.py"""
import streamlit as st

import ui

ui.page_setup("SynAudit overview")
st.caption("Live evaluation of synthetic data: quality, drift, ML utility and privacy. "
           "Open a page in the sidebar for details.")
dataset, run = ui.filters()


@st.fragment(run_every=ui.REFRESH_SECONDS)
def overview():
    q, d = ui.load("quality", dataset, run), ui.load("drift", dataset, run)
    u, p = ui.load("utility", dataset, run), ui.load("privacy", dataset, run)
    if q.empty and d.empty and u.empty and p.empty:
        st.info("InfluxDB is empty. Start the pipeline, or load demo data with "
                "`python scripts/seed_dashboard_demo.py`.")
        return
    overall = q[q["metric"] == "overall"] if not q.empty else q
    ratio = u.groupby("batch_id")["utility_ratio"].mean() if not u.empty else None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quality (overall)", f"{overall['score'].iloc[-1]:.2f}" if not overall.empty else "–",
              ui.delta(overall["score"]) if not overall.empty else None)
    c2.metric("Drift alarms", int(d["alarm"].sum()) if not d.empty else "–")
    c3.metric("Utility ratio", f"{ratio.iloc[-1]:.2f}" if ratio is not None else "–",
              ui.delta(ratio) if ratio is not None else None)
    c4.metric("Membership inference AUC", f"{p['mia_auc'].iloc[-1]:.2f}" if not p.empty else "–",
              ui.delta(p["mia_auc"]) if not p.empty else None, delta_color="inverse")
    st.divider()
    left, right = st.columns(2)
    with left:
        if not overall.empty:
            ui.line(overall, "batch_id", "score", color="dataset", title="Quality per batch", y_range=(0, 1))
        if not u.empty:
            ui.line(u, "batch_id", "utility_ratio", color="model", title="Utility ratio per batch",
                    ref=1.0, ref_label="real-data level")
    with right:
        if not d.empty:
            per = d.groupby(["dataset", "window_id"])["alarm"].sum().reset_index(name="alarms")
            ui.line(per, "window_id", "alarms", color="dataset", title="Drift alarms per window")
        if not p.empty:
            ui.line(p, "batch_id", "mia_auc", color="dataset", title="Membership inference AUC",
                    ref=0.5, ref_label="random guess", y_range=(0, 1))


overview()
