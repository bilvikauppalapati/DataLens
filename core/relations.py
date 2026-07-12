"""
relations.py — Phase 7
Finds relationships BETWEEN datasets: shared columns, likely join keys,
and correlations once two datasets are joined together.
"""
from __future__ import annotations

import pandas as pd

from core.stats import correlation_matrix


def find_shared_columns_pairwise(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    For every pair of datasets, lists column names that appear in BOTH
    (exact name match). Unlike multi_dataset.get_common_columns (which
    requires a column in ALL datasets), this works pairwise so it still
    finds relationships even if only 2 out of 5 datasets share a column.
    """
    names = list(datasets.keys())
    rows = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a, name_b = names[i], names[j]
            df_a, df_b = datasets[name_a], datasets[name_b]
            shared = set(df_a.columns) & set(df_b.columns)

            for col in sorted(shared):
                rows.append({
                    "dataset_a": name_a,
                    "dataset_b": name_b,
                    "shared_column": col,
                    "dtype_a": str(df_a[col].dtype),
                    "dtype_b": str(df_b[col].dtype),
                })

    return pd.DataFrame(rows, columns=["dataset_a", "dataset_b", "shared_column", "dtype_a", "dtype_b"])


def suggest_join_keys(
    datasets: dict[str, pd.DataFrame], overlap_threshold: float = 0.3
) -> pd.DataFrame:
    """
    For every shared column across dataset pairs, measures how much the
    ACTUAL VALUES overlap (not just the column name) — a shared name with
    zero overlapping values is a bad join key, so this ranks candidates by
    how usable they'd actually be.

    overlap_ratio = (# values common to both) / (# unique values in the
    smaller of the two columns). 1.0 means every value in the smaller
    column has a match in the other.

    Only pairs meeting overlap_threshold are returned, sorted best-first.
    """
    shared = find_shared_columns_pairwise(datasets)
    if shared.empty:
        return pd.DataFrame(columns=[
            "dataset_a", "dataset_b", "shared_column", "overlap_ratio",
            "unique_in_a", "unique_in_b", "common_values",
        ])

    rows = []
    for _, r in shared.iterrows():
        col = r["shared_column"]
        vals_a = set(datasets[r["dataset_a"]][col].dropna().unique())
        vals_b = set(datasets[r["dataset_b"]][col].dropna().unique())

        if not vals_a or not vals_b:
            continue

        common = vals_a & vals_b
        smaller_size = min(len(vals_a), len(vals_b))
        overlap_ratio = len(common) / smaller_size if smaller_size > 0 else 0.0

        if overlap_ratio >= overlap_threshold:
            rows.append({
                "dataset_a": r["dataset_a"],
                "dataset_b": r["dataset_b"],
                "shared_column": col,
                "overlap_ratio": round(overlap_ratio, 3),
                "unique_in_a": len(vals_a),
                "unique_in_b": len(vals_b),
                "common_values": len(common),
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("overlap_ratio", ascending=False).reset_index(drop=True)
    return result


def join_datasets(
    df_a: pd.DataFrame, df_b: pd.DataFrame, join_on: str, how: str = "inner"
) -> pd.DataFrame:
    """
    Joins two dataframes on a shared column. Overlapping non-join column
    names get suffixed (_a / _b) so nothing silently overwrites the other.
    """
    if join_on not in df_a.columns:
        raise ValueError(f"'{join_on}' not found in the first dataset")
    if join_on not in df_b.columns:
        raise ValueError(f"'{join_on}' not found in the second dataset")
    if how not in {"inner", "left", "right", "outer"}:
        raise ValueError(f"how must be one of inner/left/right/outer, got '{how}'")

    return pd.merge(df_a, df_b, on=join_on, how=how, suffixes=("_a", "_b"))


def cross_dataset_correlation(
    df_a: pd.DataFrame, df_b: pd.DataFrame, join_on: str, method: str = "pearson"
) -> pd.DataFrame:
    """
    Joins two datasets on a shared column, then computes the correlation
    matrix across all numeric columns of the combined result — this is
    how you find out whether a numeric column in dataset A relates to a
    numeric column in dataset B.
    """
    merged = join_datasets(df_a, df_b, join_on, how="inner")
    if merged.empty:
        raise ValueError(
            f"Joining on '{join_on}' produced no matching rows — "
            f"check that the values in this column actually overlap between the two datasets"
        )
    return correlation_matrix(merged, method=method)
