"""
multi_dataset.py — Phase 6
Small helpers for working with more than one dataset at once. This is
groundwork for Phase 7 (relationship discovery) but stays intentionally
simple here — just enough to power a side-by-side comparison view.
"""
from __future__ import annotations

import pandas as pd


def get_common_columns(datasets: dict[str, pd.DataFrame]) -> list[str]:
    """
    Returns column names that appear (by exact name match) in EVERY
    dataframe given. Useful for hinting to the user which columns might
    be worth comparing across datasets.
    """
    if not datasets:
        return []

    column_sets = [set(df.columns) for df in datasets.values()]
    common = set.intersection(*column_sets)
    return sorted(common)


def get_dataset_overview(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    One row per dataset with basic stats, for a quick side-by-side glance:
    name, row count, column count, memory usage (MB), duplicate row count.
    """
    rows = []
    for name, df in datasets.items():
        rows.append({
            "dataset": name,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3),
            "duplicate_rows": int(df.duplicated().sum()),
        })
    return pd.DataFrame(rows).set_index("dataset")
