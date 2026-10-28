import streamlit as st

import ui

ui.page_setup("Privacy risk")
st.caption("Does synthetic data disclose real training records? Exact copies, DCR and membership inference.")
dataset, run = ui.filters()


@st.fragment(run_every=ui.REFRESH_SECONDS)
def body():
    df = ui.load("privacy", dataset, run)
    if ui.empty(df, "the privacy service"):
        return
    last = df.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Exact copy rate", f"{last.get('exact_copy_rate', float('nan')):.1%}", delta_color="inverse")
    c2.metric("Median DCR", f"{last.get('median_dcr', float('nan')):.3f}")
    c3.metric("Membership inference AUC", f"{last.get('mia_auc', float('nan')):.2f}", delta_color="inverse",
              help="0.5 = attacker is guessing; higher = more risk")
    if last.get("mia_auc", 0) > 0.6 or last.get("exact_copy_rate", 0) > 0.01:
        st.warning("Latest batch shows elevated privacy risk.")
    for col, title, ref, label in [("mia_auc", "Membership inference AUC", 0.5, "random guess"),
                                   ("exact_copy_rate", "Exact copy rate", None, ""),
                                   ("median_dcr", "Median distance to closest record", None, "")]:
        if col in df.columns:
            ui.line(df, "batch_id", col, color="dataset", title=title, ref=ref, ref_label=label)
    with st.expander("Raw results"):
        st.dataframe(df, use_container_width=True, hide_index=True)


body()
