"""
session_state.py
Central place for managing Streamlit's session_state so every page
accesses datasets the same way. This dict-of-datasets pattern is what
makes multi-dataset support (Phase 6) work without a rewrite.

Structure of st.session_state.datasets:
{
    "sales.csv": {
        "df": <pd.DataFrame>,          # current (possibly transformed) version
        "original_df": <pd.DataFrame>, # untouched copy, for reset
        "transform_history": [ {"fn": "handle_missing", "args": {...}}, ... ],
        "profile": {...}               # cached output of loaders.profile_dataset
    },
    "customers.xlsx": { ... }
}
"""
from __future__ import annotations

import streamlit as st
import pandas as pd


def init_session_state() -> None:
    """Call once at the top of app/main.py."""
    if "datasets" not in st.session_state:
        st.session_state.datasets = {}
    if "active_dataset" not in st.session_state:
        st.session_state.active_dataset = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def add_dataset(name: str, df: pd.DataFrame, profile: dict) -> None:
    st.session_state.datasets[name] = {
        "df": df,
        "original_df": df.copy(),
        "transform_history": [],
        "profile": profile,
    }
    st.session_state.active_dataset = name


def get_active_df() -> pd.DataFrame | None:
    name = st.session_state.get("active_dataset")
    if name is None:
        return None
    return st.session_state.datasets[name]["df"]


def set_active_df(name: str, df: pd.DataFrame) -> None:
    st.session_state.datasets[name]["df"] = df


def list_dataset_names() -> list[str]:
    return list(st.session_state.datasets.keys())


def log_transform(name: str, fn_name: str, args: dict) -> None:
    st.session_state.datasets[name]["transform_history"].append(
        {"fn": fn_name, "args": args}
    )


def reset_dataset(name: str) -> None:
    original = st.session_state.datasets[name]["original_df"]
    st.session_state.datasets[name]["df"] = original.copy()
    st.session_state.datasets[name]["transform_history"] = []


def remove_dataset(name: str) -> None:
    del st.session_state.datasets[name]
    if st.session_state.active_dataset == name:
        remaining = list_dataset_names()
        st.session_state.active_dataset = remaining[0] if remaining else None
