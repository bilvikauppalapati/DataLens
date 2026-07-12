"""
main.py — Streamlit entrypoint.
Run with: streamlit run app/main.py
"""
import os
import sys
from pathlib import Path

# --- Crash workaround: force pyarrow to use the plain system allocator ---
# On some pyarrow/macOS builds, the default "mimalloc" allocator keeps a
# thread-local heap that isn't re-initialized correctly when Streamlit's
# ScriptRunner reuses a worker thread across reruns. That leads to a native
# SIGSEGV deep inside libarrow (mi_thread_init / mi_heap_main) whenever a
# DataFrame gets converted to an Arrow array (e.g. for st.dataframe,
# st.plotly_chart, to_parquet, etc.) — completely outside of this app's own
# Python code. Forcing the "system" memory pool avoids that code path.
# This MUST be set before pandas/pyarrow get imported anywhere.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

# Ensure project root is on sys.path so `app.*` and `core.*` imports work
# no matter what directory `streamlit run` is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.session_state import init_session_state, list_dataset_names

st.set_page_config(
    page_title="DataLens",
    page_icon="🔎",
    layout="wide",
)

init_session_state()

st.title("🔎 DataLens")
st.caption("Upload, explore, visualize, and chat with your datasets — all in one place.")

st.markdown(
    """
    ### How this app is organized
    Use the sidebar to move through each stage:

    1. **Upload & Preview** — load one or more datasets and inspect them
    2. **Preprocessing** — clean, convert types, handle missing values
    3. **Filter & Summary** — slice the data and view summary statistics
    4. **Visualize 2D** — bar, line, scatter, histogram, box, heatmap
    5. **Visualize 3D** — 3D scatter and surface plots
    6. **Relationships** — find connections across multiple datasets
    7. **Chatbot** — ask questions about your dataset(s) in plain English
    """
)

active_datasets = list_dataset_names()
if active_datasets:
    st.success(f"Loaded datasets: {', '.join(active_datasets)}")
else:
    st.info("No datasets loaded yet. Head to **Upload & Preview** in the sidebar to get started.")
