import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd

from core.stats import apply_filters, summary_stats, correlation_matrix, value_counts_for_column
from app.session_state import init_session_state, list_dataset_names

st.set_page_config(page_title="DataLens — Filter & Summary", page_icon="📊", layout="wide")
init_session_state()

st.title("📊 Filter & Summary")

dataset_names = list_dataset_names()
if not dataset_names:
    st.info("No datasets loaded yet. Go to **Upload & Preview** first.")
    st.stop()

selected = st.selectbox("Choose dataset", dataset_names)
df = st.session_state.datasets[selected]["df"]

# Filters live in their own session_state slot per dataset, so they persist
# while browsing other pages but don't touch the actual stored dataframe.
filters_key = f"_filters_{selected}"
if filters_key not in st.session_state:
    st.session_state[filters_key] = {}

st.divider()

# ---------- Filters (one column at a time) ----------
st.subheader("Filters")

filter_col = st.selectbox("Column to filter", options=list(df.columns), key="filter_col")
is_numeric = pd.api.types.is_numeric_dtype(df[filter_col])

if is_numeric:
    col_min, col_max = float(df[filter_col].min()), float(df[filter_col].max())
    chosen_range = st.slider(
        f"Range for '{filter_col}'", min_value=col_min, max_value=col_max,
        value=(col_min, col_max),
    )
    if st.button("Add / update this filter"):
        st.session_state[filters_key][filter_col] = {"min": chosen_range[0], "max": chosen_range[1]}
        st.rerun()
else:
    options = sorted(df[filter_col].dropna().unique().tolist())
    chosen_values = st.multiselect(f"Values to keep for '{filter_col}'", options=options, default=options)
    if st.button("Add / update this filter"):
        st.session_state[filters_key][filter_col] = {"values": chosen_values}
        st.rerun()

# Show active filters with a way to remove each one
active_filters = st.session_state[filters_key]
if active_filters:
    st.caption("Active filters:")
    for col_name, spec in list(active_filters.items()):
        label = f"**{col_name}**: {spec}"
        remove_col, text_col = st.columns([1, 6])
        with text_col:
            st.write(label)
        with remove_col:
            if st.button("✕", key=f"remove_{col_name}"):
                del st.session_state[filters_key][col_name]
                st.rerun()

    if st.button("Clear all filters"):
        st.session_state[filters_key] = {}
        st.rerun()

try:
    filtered_df = apply_filters(df, active_filters) if active_filters else df
except ValueError as e:
    st.error(str(e))
    filtered_df = df

st.metric("Rows after filtering", f"{len(filtered_df)} / {len(df)}")
st.dataframe(filtered_df.head(50), use_container_width=True)

st.divider()

# ---------- Summary statistics ----------
st.subheader("Summary statistics")
st.caption("Calculated on the filtered data above.")

stats_df = summary_stats(filtered_df)
st.dataframe(stats_df, use_container_width=True)

st.divider()

# ---------- Value counts for one column ----------
st.subheader("Value counts")
vc_col = st.selectbox("Column", options=list(filtered_df.columns), key="vc_col")
top_n = st.slider("Show top N values", min_value=3, max_value=30, value=10)
vc = value_counts_for_column(filtered_df, vc_col, top_n=top_n)
st.bar_chart(vc)

st.divider()

# ---------- Correlation matrix ----------
st.subheader("Correlation matrix")
numeric_cols = filtered_df.select_dtypes(include="number").columns.tolist()

if len(numeric_cols) < 2:
    st.info("Need at least 2 numeric columns to compute correlations.")
else:
    method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
    corr = correlation_matrix(filtered_df, method=method)
    st.dataframe(corr.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1), use_container_width=True)
