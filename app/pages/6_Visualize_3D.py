import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from core.viz import scatter_3d, surface_3d
from app.session_state import init_session_state, list_dataset_names

st.set_page_config(page_title="DataLens — Visualize 3D", page_icon="🧊", layout="wide")
init_session_state()

st.title("🧊 Visualize 3D")

dataset_names = list_dataset_names()
if not dataset_names:
    st.info("No datasets loaded yet. Go to **Upload & Preview** first.")
    st.stop()

selected = st.selectbox("Choose dataset", dataset_names)
df = st.session_state.datasets[selected]["df"]

# Reuse filters from Filter & Summary page if any are set
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

numeric_cols = chart_df.select_dtypes(include="number").columns.tolist()

if len(numeric_cols) < 3:
    st.warning("3D visualization needs at least 3 numeric columns. This dataset doesn't have enough.")
    st.stop()

st.divider()

chart_type = st.radio(
    "Chart type", ["3D Scatter", "3D Surface"], horizontal=True,
    help="Scatter plots each row as a point. Surface needs your X/Y columns to form a grid "
         "(e.g. repeated coordinates) — z values are averaged where they overlap.",
)

col_x, col_y, col_z = st.columns(3)
with col_x:
    x_col = st.selectbox("X axis", options=numeric_cols, key="x3d")
with col_y:
    y_col = st.selectbox("Y axis", options=[c for c in numeric_cols if c != x_col], key="y3d")
with col_z:
    z_col = st.selectbox("Z axis", options=[c for c in numeric_cols if c not in (x_col, y_col)], key="z3d")

try:
    if chart_type == "3D Scatter":
        non_numeric_cols = [c for c in chart_df.columns if c not in numeric_cols]
        color_options = ["(none)"] + non_numeric_cols
        color_choice = st.selectbox("Color grouping (optional)", options=color_options)
        color_col = None if color_choice == "(none)" else color_choice

        fig = scatter_3d(chart_df, x_col, y_col, z_col, color=color_col)
    else:
        fig = surface_3d(chart_df, x_col, y_col, z_col)

    st.plotly_chart(fig, use_container_width=True)

except ValueError as e:
    st.error(str(e))
