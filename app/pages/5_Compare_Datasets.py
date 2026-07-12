import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from core.multi_dataset import get_common_columns, get_dataset_overview
from core.stats import summary_stats
from core.viz import suggest_chart_type, bar_chart, histogram, scatter_chart
from app.session_state import init_session_state, list_dataset_names

st.set_page_config(page_title="DataLens — Compare Datasets", page_icon="🗂️", layout="wide")
init_session_state()

st.title("🗂️ Compare Datasets")

dataset_names = list_dataset_names()
if len(dataset_names) < 2:
    st.info("Load at least 2 datasets on the **Upload & Preview** page to use this comparison view.")
    st.stop()

selected = st.multiselect(
    "Choose 2 or more datasets to compare", options=dataset_names, default=dataset_names[:2]
)

if len(selected) < 2:
    st.warning("Select at least 2 datasets.")
    st.stop()

selected_dfs = {name: st.session_state.datasets[name]["df"] for name in selected}

st.divider()

# ---------- Overview table ----------
st.subheader("Overview")
overview = get_dataset_overview(selected_dfs)
st.dataframe(overview, use_container_width=True)

common_cols = get_common_columns(selected_dfs)
if common_cols:
    st.success(f"Columns shared by all selected datasets: {', '.join(common_cols)}")
else:
    st.info("No columns with the exact same name are shared across all selected datasets.")

st.divider()

# ---------- Side-by-side preview ----------
st.subheader("Preview")
cols = st.columns(len(selected))
for col, name in zip(cols, selected):
    with col:
        st.caption(name)
        st.dataframe(selected_dfs[name].head(10), use_container_width=True)

st.divider()

# ---------- Side-by-side summary stats ----------
st.subheader("Summary statistics")
cols = st.columns(len(selected))
for col, name in zip(cols, selected):
    with col:
        st.caption(name)
        st.dataframe(summary_stats(selected_dfs[name]), use_container_width=True)

st.divider()

# ---------- Side-by-side chart on a shared column ----------
st.subheader("Compare a column across datasets")

if common_cols:
    compare_col = st.selectbox("Column to compare (shared across all selected datasets)", options=common_cols)

    cols = st.columns(len(selected))
    for col, name in zip(cols, selected):
        with col:
            st.caption(name)
            df = selected_dfs[name]
            try:
                chart_type = suggest_chart_type(df, compare_col)
                if chart_type == "histogram":
                    fig = histogram(df, compare_col)
                else:
                    fig = bar_chart(df, compare_col)
                st.plotly_chart(fig, use_container_width=True)
            except ValueError as e:
                st.error(str(e))
else:
    st.info("No shared columns to compare — pick datasets with at least one matching column name.")
