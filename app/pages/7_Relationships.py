import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from core.relations import (
    find_shared_columns_pairwise, suggest_join_keys, join_datasets, cross_dataset_correlation,
)
from core.viz import heatmap
from app.session_state import init_session_state, list_dataset_names

st.set_page_config(page_title="DataLens — Relationships", page_icon="🔗", layout="wide")
init_session_state()

st.title("🔗 Relationships")
st.caption("Find shared columns, good join keys, and correlations between two datasets.")

dataset_names = list_dataset_names()
if len(dataset_names) < 2:
    st.info("Load at least 2 datasets on the **Upload & Preview** page to discover relationships.")
    st.stop()

all_dfs = {name: st.session_state.datasets[name]["df"] for name in dataset_names}

st.divider()

# ---------- Shared columns across ALL loaded datasets ----------
st.subheader("Shared columns")
shared = find_shared_columns_pairwise(all_dfs)

if shared.empty:
    st.info("No two loaded datasets share a column with the exact same name.")
else:
    st.dataframe(shared, use_container_width=True)

st.divider()

# ---------- Suggested join keys ----------
st.subheader("Suggested join keys")
st.caption(
    "Ranks shared columns by how much their actual VALUES overlap — a shared "
    "name with no matching values makes a bad join key, so this checks the real data."
)

threshold = st.slider("Minimum value overlap to show", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
join_candidates = suggest_join_keys(all_dfs, overlap_threshold=threshold)

if join_candidates.empty:
    st.info("No column pairs meet this overlap threshold. Try lowering the slider.")
else:
    st.dataframe(join_candidates, use_container_width=True)

st.divider()

# ---------- Join two datasets + cross-dataset correlation ----------
st.subheader("Join two datasets & find correlations")

col_a, col_b = st.columns(2)
with col_a:
    dataset_a = st.selectbox("First dataset", options=dataset_names, key="rel_a")
with col_b:
    other_options = [n for n in dataset_names if n != dataset_a]
    dataset_b = st.selectbox("Second dataset", options=other_options, key="rel_b")

df_a, df_b = all_dfs[dataset_a], all_dfs[dataset_b]
shared_cols = sorted(set(df_a.columns) & set(df_b.columns))

if not shared_cols:
    st.warning(f"'{dataset_a}' and '{dataset_b}' don't share any column names, so they can't be joined directly.")
else:
    join_col = st.selectbox("Join on column", options=shared_cols)
    how = st.radio("Join type", ["inner", "left", "right", "outer"], horizontal=True,
                    help="'inner' keeps only rows that match in both datasets — usually what you want.")

    if st.button("Join & analyze", type="primary"):
        try:
            merged = join_datasets(df_a, df_b, join_col, how=how)
            st.success(f"Joined into {len(merged)} rows.")
            st.dataframe(merged.head(50), use_container_width=True)

            numeric_cols = merged.select_dtypes(include="number").columns.tolist()
            if len(numeric_cols) >= 2:
                st.subheader("Correlation across the joined data")
                corr = cross_dataset_correlation(df_a, df_b, join_col)
                fig = heatmap(corr)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns in the joined result to show correlations.")

        except ValueError as e:
            st.error(str(e))
