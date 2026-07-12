"""
lookup.py — Phase 8 support module
Deterministic, exact row/value lookups — NO LLM involved in finding the
actual answer. The chatbot may help figure out WHAT the user is asking
for, but the real lookup always runs as plain pandas code here, so the
result is guaranteed accurate rather than something the model guessed.
"""
from __future__ import annotations

import pandas as pd

from core.relations import join_datasets


def _normalize(series: pd.Series) -> pd.Series:
    """Case/whitespace-insensitive string form of a column, used for matching
    so 'India', ' india ', and 'INDIA' are all treated as the same value."""
    return series.astype(str).str.strip().str.lower()


def find_matching_rows(df: pd.DataFrame, column: str, value) -> pd.DataFrame:
    """
    Returns all rows where `column` matches `value` (case/whitespace
    insensitive for text; also works for numbers since they're compared
    as their string form).
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    target = str(value).strip().lower()
    return df[_normalize(df[column]) == target]

def search_value_in_datasets(value: str, datasets: dict[str, pd.DataFrame]) -> list[dict]:
    """
    Searches EVERY column of EVERY given dataset for an exact match of
    `value` (case/whitespace insensitive). This is a real, direct search
    over the actual data — not something an LLM guesses at — so it finds
    a matching row regardless of dataset size, unlike a small text sample.

    Returns a list of {"dataset": name, "column": col, "rows": DataFrame}
    for every (dataset, column) combination where at least one row matches.
    An empty list means the value genuinely doesn't appear anywhere.
    """
    results = []
    for dataset_name, df in datasets.items():
        for column in df.columns:
            try:
                matches = find_matching_rows(df, column, value)
            except ValueError:
                continue
            if not matches.empty:
                results.append({"dataset": dataset_name, "column": column, "rows": matches})
    return results

def cross_dataset_lookup(
    df_a: pd.DataFrame, df_b: pd.DataFrame, join_column: str, lookup_value
) -> pd.DataFrame:
    """
    Finds the row(s) in df_a matching `lookup_value` on `join_column`,
    then returns the corresponding row(s) in df_b that share the same
    join_column value — i.e. "what's in table B for this same key".

    Raises ValueError with a clear message if the value isn't found in
    either dataset, rather than silently returning nothing.
    """
    if join_column not in df_a.columns:
        raise ValueError(f"'{join_column}' not found in the first dataset")
    if join_column not in df_b.columns:
        raise ValueError(f"'{join_column}' not found in the second dataset")

    matches_a = find_matching_rows(df_a, join_column, lookup_value)
    if matches_a.empty:
        raise ValueError(
            f"No row found in the first dataset where '{join_column}' matches '{lookup_value}'"
        )

    matched_keys = _normalize(matches_a[join_column]).unique().tolist()
    result = df_b[_normalize(df_b[join_column]).isin(matched_keys)]

    if result.empty:
        raise ValueError(
            f"Found '{lookup_value}' in the first dataset, but no matching "
            f"'{join_column}' value in the second dataset"
        )

    return result