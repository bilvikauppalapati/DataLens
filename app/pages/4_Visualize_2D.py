import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd

from core.viz import (
    suggest_chart_type, bar_chart, line_chart, scatter_chart, histogram, box_plot, heatmap,
)
from core.stats import correlation_matrix
from app.session_state import init_session_state, list_dataset_names

st.set_page_config(page_title="DataLens — Visualize 2D", page_icon="📈", layout="wide")
init_session_state()

st.title("📈 Visualize 2D")

dataset_names = list_dataset_names()
if not dataset_names:
    st.info("No datasets loaded yet. Go to **Upload & Preview** first.")
    st.stop()

selected = st.selectbox("Choose dataset", dataset_names)
df = st.session_state.datasets[selected]["df"]

# Reuse filters set on the Filter & Summary page if they exist, so charts
# reflect the same filtered view the user was just looking at.
filters_key = f"_filters_{selected}"
if filters_key in st.session_state and st.session_state[filters_key]:
    from core.stats import apply_filters
    try:
        chart_df = apply_filters(df, st.session_state[filters_key])
        st.caption(f"Using filtered data from Filter & Summary page ({len(chart_df)} / {len(df)} rows).")
    except ValueError:
        chart_df = df
else:
    chart_df = df

st.divider()

columns = list(chart_df.columns)

col_x, col_y = st.columns(2)
with col_x:
    x_col = st.selectbox("X axis (required)", options=columns, key="x_col")
with col_y:
    y_options = ["(none)"] + columns
    y_col_raw = st.selectbox("Y axis (optional)", options=y_options, key="y_col")
    y_col = None if y_col_raw == "(none)" else y_col_raw

color_options = ["(none)"] + [c for c in columns if c not in (x_col, y_col)]
color_col_raw = st.selectbox("Color grouping (optional)", options=color_options, key="color_col")
color_col = None if color_col_raw == "(none)" else color_col_raw

suggested = suggest_chart_type(chart_df, x_col, y_col)
chart_type_options = ["bar", "line", "scatter", "histogram", "box", "heatmap"]
default_index = chart_type_options.index(suggested) if suggested in chart_type_options else 0

chart_type = st.selectbox(
    "Chart type", options=chart_type_options, index=default_index,
    help=f"Auto-suggested based on your column types: '{suggested}'. Change if you'd like a different view.",
)

st.divider()

try:
    if chart_type == "bar":
        fig = bar_chart(chart_df, x_col, y_col, color=color_col)
    elif chart_type == "line":
        if y_col is None:
            st.warning("Line chart needs a Y axis column.")
            st.stop()
        fig = line_chart(chart_df, x_col, y_col, color=color_col)
    elif chart_type == "scatter":
        if y_col is None:
            st.warning("Scatter chart needs a Y axis column.")
            st.stop()
        fig = scatter_chart(chart_df, x_col, y_col, color=color_col)
    elif chart_type == "histogram":
        fig = histogram(chart_df, x_col)
    elif chart_type == "box":
        if y_col is None:
            st.warning("Box plot needs a Y axis column (numeric values), with X as the grouping column.")
            st.stop()
        fig = box_plot(chart_df, x_col, y_col)
    elif chart_type == "heatmap":
        numeric_cols = chart_df.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for a correlation heatmap.")
            st.stop()
        corr = correlation_matrix(chart_df)
        fig = heatmap(corr)
    else:
        st.error(f"Unknown chart type: {chart_type}")
        st.stop()

    st.plotly_chart(fig, use_container_width=True)

except ValueError as e:
    st.error(str(e))
