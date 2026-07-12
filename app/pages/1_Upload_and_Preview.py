import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from core.loaders import load_dataset, profile_dataset
from app.session_state import init_session_state, add_dataset, list_dataset_names

st.set_page_config(page_title="DataLens — Upload & Preview", page_icon="📤", layout="wide")
init_session_state()

st.title("📤 Upload & Preview")

uploaded_files = st.file_uploader(
    "Upload one or more datasets",
    type=["csv", "xlsx", "xls", "json", "parquet"],
    accept_multiple_files=True,
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name in list_dataset_names():
            continue  # already loaded this session
        try:
            result = load_dataset(uploaded_file.getvalue(), uploaded_file.name)
            profile = profile_dataset(result.dataframe)
            add_dataset(uploaded_file.name, result.dataframe, profile)
            for w in result.warnings:
                st.warning(f"{uploaded_file.name}: {w}")
            st.success(f"Loaded {uploaded_file.name}")
        except ValueError as e:
            st.error(f"Failed to load {uploaded_file.name}: {e}")

dataset_names = list_dataset_names()

if not dataset_names:
    st.info("Upload a file above to see a preview here.")
else:
    selected = st.selectbox("Preview dataset", dataset_names)
    data = st.session_state.datasets[selected]
    df = data["df"]
    profile = data["profile"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", profile["n_rows"])
    col2.metric("Columns", profile["n_cols"])
    col3.metric("Duplicate rows", profile["duplicate_rows"])
    col4.metric("Memory (MB)", profile["memory_usage_mb"])

    st.subheader("Preview")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Column info")
    st.dataframe(
        {
            "dtype": profile["dtypes"],
            "nulls": profile["null_counts"],
            "null %": profile["null_pct"],
        },
        use_container_width=True,
    )
