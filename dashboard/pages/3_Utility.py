import streamlit as st

import ui

ui.page_setup("Machine learning utility")
st.caption("Train on synthetic, test on real (TSTR) vs train on real (TRTR). Ratio = TSTR AUC / TRTR AUC.")
dataset, run = ui.filters()


@st.fragment(run_every=ui.REFRESH_SECONDS)
def body():
    df = ui.load("utility", dataset, run)
    if ui.empty(df, "the utility service"):
        return
    latest = df[df["batch_id"] == df["batch_id"].max()]
    cols = st.columns(max(1, latest["model"].nunique()))
    for col, (model, grp) in zip(cols, latest.groupby("model")):
        col.metric(f"{model} utility ratio", f"{grp['utility_ratio'].mean():.2f}",
                   ui.delta(df[df["model"] == model]["utility_ratio"]))
    ui.line(df, "batch_id", "utility_ratio", color="model", title="Utility ratio per batch",
            ref=1.0, ref_label="real-data level")
    metric = st.radio("Detail metric", [m for m in ("roc_auc", "f1", "accuracy") if m in df.columns], horizontal=True)
    if metric:
        ui.line(df, "batch_id", metric, color="model", title=f"TSTR {metric} per batch", y_range=(0, 1))
    with st.expander("Raw results"):
        st.dataframe(df, use_container_width=True, hide_index=True)


body()
