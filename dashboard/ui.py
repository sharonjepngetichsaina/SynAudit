"""Shared helpers for the Streamlit pages: filters, cached loading, charts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `storage`

import pandas as pd
import plotly.express as px
import streamlit as st

from storage import db

REFRESH_SECONDS = 5


def page_setup(title: str) -> None:
    st.set_page_config(page_title=f"SynAudit · {title}", layout="wide")
    st.title(title)


def filters() -> tuple[str | None, str | None]:
    """Sidebar dataset / run selectors, remembered across pages."""
    st.sidebar.header("Filters")
    try:
        datasets, runs = db.tag_values("dataset"), db.tag_values("run_id")
    except Exception as exc:
        st.error(f"Cannot reach InfluxDB: {exc}")
        st.stop()
    return _select("Dataset", datasets, "f_dataset"), _select("Run / experiment", runs, "f_run")


def _select(label, options, key):
    choices = ["All"] + list(options)
    if st.session_state.get(key) not in choices:
        st.session_state.pop(key, None)
    value = st.sidebar.selectbox(label, choices, key=key)
    return None if value == "All" else value


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def load(measurement: str, dataset: str | None, run: str | None) -> pd.DataFrame:
    return db.read(measurement, dataset=dataset, run=run)


def empty(df: pd.DataFrame, who: str) -> bool:
    if df.empty:
        st.info(f"No results yet. They appear here as soon as {who} writes to InfluxDB.")
        return True
    return False


def line(df, x, y, color=None, title="", ref=None, ref_label="", y_range=None):
    fig = px.line(df, x=x, y=y, color=color, markers=True, title=title)
    if ref is not None:
        fig.add_hline(y=ref, line_dash="dash", line_color="gray",
                      annotation_text=ref_label, annotation_position="top left")
    if y_range:
        fig.update_yaxes(range=list(y_range))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)


def delta(series: pd.Series):
    s = series.dropna()
    return None if len(s) < 2 else round(float(s.iloc[-1] - s.iloc[-2]), 3)
