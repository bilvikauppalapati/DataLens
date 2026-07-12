"""
stats.py — Phase 3
Filtering and summary statistics. Every function takes a df and returns
a NEW df/Series (no in-place mutation), consistent with core/preprocessing.py.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Apply one or more column filters to a dataframe.

    filters format — a dict keyed by column name, where each value is one of:
        {"min": 18, "max": 65}            -> numeric range filter (either key optional)
        {"values": ["US", "IN"]}          -> categorical "keep only these values" filter

    Example:
        apply_filters(df, {
            "age": {"min": 18, "max": 65},
            "country": {"values": ["US", "IN"]},
        })
    """
    result = df.copy()

    for column, spec in filters.items():
        if column not in result.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")

        if "values" in spec:
            result = result[result[column].isin(spec["values"])]

        elif "min" in spec or "max" in spec:
            if not pd.api.types.is_numeric_dtype(result[column]):
                raise ValueError(f"min/max filter requires a numeric column; '{column}' is not numeric")
            if "min" in spec and spec["min"] is not None:
                result = result[result[column] >= spec["min"]]
            if "max" in spec and spec["max"] is not None:
                result = result[result[column] <= spec["max"]]

        else:
            raise ValueError(f"Filter spec for '{column}' must contain 'values' or 'min'/'max'")

    return result.reset_index(drop=True)


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per column with relevant summary statistics.
    Numeric columns get: count, mean, std, min, 25%, 50%, 75%, max, skew, kurtosis.
    Non-numeric columns get: count, unique, top (most frequent value), freq.
    """
    rows = []
    for col in df.columns:
        series = df[col]
        row = {"column": col, "dtype": str(series.dtype), "count": int(series.count())}

        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            row.update({
                "mean": desc.get("mean"),
                "std": desc.get("std"),
                "min": desc.get("min"),
                "25%": desc.get("25%"),
                "50%": desc.get("50%"),
                "75%": desc.get("75%"),
                "max": desc.get("max"),
                "skew": series.skew(),
                "kurtosis": series.kurtosis(),
                "unique": None,
                "top": None,
                "freq": None,
            })
        else:
            value_counts = series.value_counts()
            row.update({
                "mean": None, "std": None, "min": None, "25%": None,
                "50%": None, "75%": None, "max": None, "skew": None, "kurtosis": None,
                "unique": int(series.nunique()),
                "top": value_counts.index[0] if len(value_counts) > 0 else None,
                "freq": int(value_counts.iloc[0]) if len(value_counts) > 0 else None,
            })

        rows.append(row)

    return pd.DataFrame(rows).set_index("column")


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """
    Correlation matrix across numeric columns only.
    method: "pearson" (linear correlation) or "spearman" (rank correlation,
    more robust to outliers and non-linear monotonic relationships).
    """
    if method not in {"pearson", "spearman"}:
        raise ValueError(f"method must be 'pearson' or 'spearman', got '{method}'")

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns to compute a correlation matrix")

    return numeric_df.corr(method=method)


def value_counts_for_column(df: pd.DataFrame, column: str, top_n: int = 10) -> pd.Series:
    """Top N most frequent values in a column, as a Series (value -> count)."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    return df[column].value_counts().head(top_n)
