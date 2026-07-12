import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd

from core.preprocessing import (
    handle_missing, smart_fill_missing, convert_dtype, get_compatible_dtypes,
    get_source_dtype_group, drop_duplicates, detect_outliers,
)
from app.session_state import (
    init_session_state, list_dataset_names, set_active_df, log_transform, reset_dataset,
)

st.set_page_config(page_title="DataLens — Preprocessing", page_icon="🧹", layout="wide")
init_session_state()

st.title("🧹 Preprocessing")

dataset_names = list_dataset_names()
if not dataset_names:
    st.info("No datasets loaded yet. Go to **Upload & Preview** first.")
    st.stop()

selected = st.selectbox("Choose dataset to clean", dataset_names)
data = st.session_state.datasets[selected]
df = data["df"]

col_reset, col_hist = st.columns([1, 3])
with col_reset:
    if st.button("↩️ Reset to original"):
        reset_dataset(selected)
        st.rerun()
with col_hist:
    if data["transform_history"]:
        with st.expander(f"Transform history ({len(data['transform_history'])} steps)"):
            for i, step in enumerate(data["transform_history"], 1):
                st.text(f"{i}. {step['fn']}({step['args']})")

st.divider()

# ---------- Missing values (one column at a time) ----------
st.subheader("Missing values")
null_counts = df.isnull().sum()
cols_with_nulls = null_counts[null_counts > 0]

if cols_with_nulls.empty:
    st.success("No missing values in this dataset.")
else:
    st.dataframe(cols_with_nulls.rename("null count"), use_container_width=True)

    target_col = st.selectbox("Column to fix", options=list(cols_with_nulls.index))
    is_numeric = pd.api.types.is_numeric_dtype(df[target_col])

    action = st.radio("What do you want to do?", ["Fill values", "Remove"], horizontal=True)

    if action == "Fill values":
        if is_numeric:
            numeric_strategy = st.selectbox("Fill using:", ["mean", "median", "mode"])
        else:
            numeric_strategy = "mode"
            st.caption("This is a non-numeric column, so it will be filled using **mode** (most common value).")

        if st.button("Apply fill", type="primary"):
            try:
                new_df = smart_fill_missing(df, columns=[target_col], numeric_strategy=numeric_strategy)
                set_active_df(selected, new_df)
                log_transform(
                    selected, "smart_fill_missing",
                    {"column": target_col, "numeric_strategy": numeric_strategy},
                )
                st.success(f"Filled missing values in '{target_col}'")
                st.rerun()
            except ValueError as e:
                st.error(str(e))

    else:  # Remove
        remove_mode = st.radio(
            "Remove:", ["Rows with missing values", "This column entirely"], horizontal=True
        )
        if st.button("Apply removal", type="primary"):
            strategy = "drop_rows" if remove_mode == "Rows with missing values" else "drop_cols"
            new_df = handle_missing(df, strategy, columns=[target_col])
            set_active_df(selected, new_df)
            log_transform(selected, "handle_missing", {"strategy": strategy, "column": target_col})
            st.success(f"Applied '{strategy}' to '{target_col}'")
            st.rerun()

st.divider()

# ---------- Dtype conversion (one column at a time, compatibility-filtered) ----------
st.subheader("Convert column type")
st.caption("Supported types: text, integer, float (decimal point), and precise decimal.")

dtype_col = st.selectbox("Column", options=list(df.columns), key="dtype_col")
current_dtype_group = get_source_dtype_group(df, dtype_col)
compatible_targets = get_compatible_dtypes(df, dtype_col)

st.caption(f"Current type: **{current_dtype_group}**")

if not compatible_targets:
    st.info("No sensible conversions available for this column's current type.")
else:
    target_dtype = st.selectbox(
        "Convert to", options=compatible_targets, key="target_dtype",
        help="Only conversions that make sense for this column's current type are shown."
    )

    if st.button("Apply type conversion"):
        try:
            new_df = convert_dtype(df, dtype_col, target_dtype)
            set_active_df(selected, new_df)
            log_transform(
                selected, "convert_dtype",
                {"column": dtype_col, "target_dtype": target_dtype},
            )
            st.success(f"Converted '{dtype_col}' to {target_dtype}")
            st.rerun()
        except ValueError as e:
            st.error(str(e))

st.divider()

# ---------- Duplicates ----------
st.subheader("Duplicate rows")
dup_count = df.duplicated().sum()
st.metric("Duplicate rows found", int(dup_count))

if dup_count > 0 and st.button("Remove duplicate rows"):
    new_df = drop_duplicates(df)
    set_active_df(selected, new_df)
    log_transform(selected, "drop_duplicates", {})
    st.success(f"Removed {dup_count} duplicate rows")
    st.rerun()

st.divider()

# ---------- Outliers (one column at a time) ----------
st.subheader("Outlier detection")
numeric_cols = df.select_dtypes(include="number").columns.tolist()

if not numeric_cols:
    st.info("No numeric columns to check for outliers.")
else:
    outlier_col = st.selectbox("Column", options=numeric_cols, key="outlier_col")
    method = st.selectbox("Method", ["iqr", "zscore"], key="outlier_method")

    if st.button("Detect outliers"):
        try:
            mask = detect_outliers(df, outlier_col, method=method)
            n_outliers = int(mask.sum())
            st.session_state["_outlier_mask"] = mask
            st.session_state["_outlier_col"] = outlier_col
            st.write(f"Found **{n_outliers}** outlier rows out of {len(df)}")
            if n_outliers > 0:
                st.dataframe(df[mask], use_container_width=True)
        except ValueError as e:
            st.error(str(e))

    # Show removal button if a detection was already run for this column
    if st.session_state.get("_outlier_col") == outlier_col:
        mask = st.session_state.get("_outlier_mask")
        if mask is not None and mask.sum() > 0:
            if st.button(f"Remove these {int(mask.sum())} outlier rows"):
                new_df = df[~mask].reset_index(drop=True)
                set_active_df(selected, new_df)
                log_transform(selected, "detect_outliers_removed", {"column": outlier_col, "method": method})
                st.session_state.pop("_outlier_mask", None)
                st.session_state.pop("_outlier_col", None)
                st.success("Outlier rows removed")
                st.rerun()

st.divider()
st.subheader("Current data preview")
st.dataframe(df.head(50), use_container_width=True)
