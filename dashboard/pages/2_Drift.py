import plotly.express as px
import streamlit as st

import ui

ui.page_setup("Drift detection")
st.caption("Distribution changes in the synthetic stream per window (KS, chi-square, ADWIN, Page-Hinkley).")
dataset, run = ui.filters()
alpha = st.sidebar.slider("Significance level", 0.001, 0.10, 0.05, 0.001, format="%.3f")


@st.fragment(run_every=ui.REFRESH_SECONDS)
def body():
    df = ui.load("drift", dataset, run)
    if ui.empty(df, "the drift service"):
        return
    df["alarm"] = df["alarm"].astype(bool)
    alarms = df[df["alarm"]]
    c1, c2, c3 = st.columns(3)
    c1.metric("Windows analysed", int(df["window_id"].nunique()))
    c2.metric("Alarms", len(alarms))
    c3.metric("First alarm at window", int(alarms["window_id"].min()) if not alarms.empty else "none")
    if "p_value" in df.columns:
        tests = df.dropna(subset=["p_value"])
        if not tests.empty:
            cols = sorted(tests["column"].unique())
            chosen = st.multiselect("Columns", cols, default=cols[:6])
            shown = tests[tests["column"].isin(chosen)]
            if not shown.empty:
                ui.line(shown, "window_id", "p_value", color="column",
                        title="p-value per window (below the line = significant change)",
                        ref=alpha, ref_label=f"α = {alpha}", y_range=(0, 1))
    heat = df.groupby(["column", "window_id"])["alarm"].max().astype(int).reset_index()
    pivot = heat.pivot(index="column", columns="window_id", values="alarm").fillna(0)
    fig = px.imshow(pivot, color_continuous_scale=["#e8eef5", "#d1495b"], aspect="auto",
                    title="Alarm map (red = alarm)")
    fig.update_layout(coloraxis_showscale=False, height=120 + 22 * len(pivot))
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Alarm log")
    if alarms.empty:
        st.success("No drift detected.")
    else:
        show = [c for c in ("time", "dataset", "window_id", "column", "test", "statistic", "p_value") if c in alarms]
        st.dataframe(alarms.sort_values("time", ascending=False)[show], use_container_width=True, hide_index=True)


body()
