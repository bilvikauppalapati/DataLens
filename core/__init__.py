"""
core/ — all dataset logic lives here, completely independent of Streamlit.

Rule for this package: NOTHING in core/ should import streamlit.
This keeps the logic reusable if we later swap the UI for FastAPI + React,
or expose it as a CLI / API.
"""
