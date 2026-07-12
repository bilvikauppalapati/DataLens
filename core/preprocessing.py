"""
preprocessing.py — Phase 2
Cleaning and transformation functions. Every function takes a df and
returns a NEW df (no in-place mutation) — the caller (Streamlit page)
decides whether to keep or discard the result, and logs it to
transform_history in session_state.
"""
from __future__ import annotations

import decimal

import pandas as pd
import numpy as np


VALID_MISSING_STRATEGIES = {
    "drop_rows", "drop_cols", "mean", "median", "mode", "ffill", "bfill", "constant"
}

# Only these 4 types are supported for conversion. bool, datetime, category,
# and complex were deliberately dropped: bool in particular silently destroys
# data (Python treats almost any non-empty string as True), and the others
# added complexity that isn't needed for this project's datasets.
VALID_DTYPES = {"str", "int", "float", "decimal"}

# All 4 types can convert to any of the others — str/int/float/decimal are
# all scalar, single-value representations of the same underlying data,
# so every combination is safe (the actual cell values decide if it succeeds).
DTYPE_COMPATIBILITY = {
    "str": ["int", "float", "decimal"],
    "int": ["str", "float", "decimal"],
    "float": ["str", "int", "decimal"],
    "decimal": ["str", "int", "float"],
}


def _is_decimal_column(col: "pd.Series") -> bool:
    """Decimal columns are stored as plain object columns holding Decimal
    instances (pandas has no native decimal dtype), so we detect them by
    checking the actual value type instead of col.dtype."""
    non_null = col.dropna()
    if len(non_null) == 0:
        return False
    return isinstance(non_null.iloc[0], decimal.Decimal)


def get_source_dtype_group(df: "pd.DataFrame", column: str) -> str:
    """Classify a column's current dtype into one of the DTYPE_COMPATIBILITY keys."""
    col = df[column]
    if _is_decimal_column(col):
        return "decimal"
    elif pd.api.types.is_integer_dtype(col):
        return "int"
    elif pd.api.types.is_float_dtype(col):
        return "float"
    else:
        return "str"


def get_compatible_dtypes(df: "pd.DataFrame", column: str) -> list[str]:
    """Returns the list of target dtypes that make sense for this column right now."""
    source_group = get_source_dtype_group(df, column)
    return DTYPE_COMPATIBILITY.get(source_group, list(VALID_DTYPES))


def smart_fill_missing(
    df: pd.DataFrame,
    columns: list[str],
    numeric_strategy: str = "mean",
) -> pd.DataFrame:
    """
    Fill missing values without the caller needing to know column types.

    - Numeric columns in `columns` are filled using `numeric_strategy`
      (must be "mean", "median", or "mode").
    - Non-numeric columns in `columns` are always filled using "mode",
      since mean/median don't make sense for text/categorical data.

    This removes the "mean requires numeric column" failure mode entirely —
    the caller just says which columns to fix and how to treat numbers,
    and this function does the right thing for everything else.
    """
    if numeric_strategy not in {"mean", "median", "mode"}:
        raise ValueError(f"numeric_strategy must be 'mean', 'median', or 'mode', got '{numeric_strategy}'")

    result = df.copy()
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(result[c])]
    non_numeric_cols = [c for c in columns if c not in numeric_cols]

    if numeric_cols:
        result = handle_missing(result, numeric_strategy, columns=numeric_cols)
    if non_numeric_cols:
        result = handle_missing(result, "mode", columns=non_numeric_cols)

    return result


def handle_missing(
    df: pd.DataFrame,
    strategy: str,
    columns: list[str] | None = None,
    constant_value=None,
) -> pd.DataFrame:
    """
    Handle missing values using the given strategy.

    Args:
        df: input dataframe
        strategy: one of VALID_MISSING_STRATEGIES
        columns: which columns to apply to. None = all columns.
        constant_value: required only when strategy == "constant"
    """
    if strategy not in VALID_MISSING_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Must be one of {VALID_MISSING_STRATEGIES}")

    result = df.copy()
    target_cols = columns if columns else list(result.columns)

    if strategy == "drop_rows":
        return result.dropna(subset=target_cols)

    if strategy == "drop_cols":
        return result.drop(columns=target_cols)

    for col in target_cols:
        if col not in result.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")

        if strategy == "mean":
            if not pd.api.types.is_numeric_dtype(result[col]):
                raise ValueError(f"'mean' strategy requires a numeric column; '{col}' is not numeric")
            result[col] = result[col].fillna(result[col].mean())

        elif strategy == "median":
            if not pd.api.types.is_numeric_dtype(result[col]):
                raise ValueError(f"'median' strategy requires a numeric column; '{col}' is not numeric")
            result[col] = result[col].fillna(result[col].median())

        elif strategy == "mode":
            mode_vals = result[col].mode()
            if len(mode_vals) > 0:
                result[col] = result[col].fillna(mode_vals.iloc[0])

        elif strategy == "ffill":
            result[col] = result[col].ffill()

        elif strategy == "bfill":
            result[col] = result[col].bfill()

        elif strategy == "constant":
            if constant_value is None:
                raise ValueError("'constant' strategy requires constant_value to be set")
            result[col] = result[col].fillna(constant_value)

    return result


def convert_dtype(df: pd.DataFrame, column: str, target_dtype: str) -> pd.DataFrame:
    """
    Convert a single column to a new dtype. Raises a clear error on failure.
    Supports only: "str", "int", "float", "decimal".
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    if target_dtype not in VALID_DTYPES:
        raise ValueError(f"target_dtype must be one of {VALID_DTYPES}, got '{target_dtype}'")

    result = df.copy()
    try:
        if target_dtype == "int":
            result[column] = pd.to_numeric(result[column], errors="raise").astype("Int64")
        elif target_dtype == "float":
            result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
        elif target_dtype == "str":
            result[column] = result[column].astype(str)
        elif target_dtype == "decimal":
            # Convert via str() first, not directly from float, to avoid baking in
            # float rounding artifacts (e.g. Decimal(0.1) != Decimal("0.1")).
            def _to_decimal(value):
                if pd.isna(value):
                    return value
                return decimal.Decimal(str(value))
            result[column] = result[column].apply(_to_decimal)
    except (ValueError, TypeError, decimal.InvalidOperation) as e:
        raise ValueError(
            f"Could not convert column '{column}' to {target_dtype}: {e}. "
            f"Check for non-{target_dtype}-compatible values in this column."
        ) from e

    return result



def drop_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    """Remove duplicate rows. subset=None checks all columns."""
    return df.drop_duplicates(subset=subset).reset_index(drop=True)


def detect_outliers(df: pd.DataFrame, column: str, method: str = "iqr") -> pd.Series:
    """
    Returns a boolean Series (same index as df) marking True for outlier rows.
    Does NOT remove anything — caller decides what to do with the flags.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Outlier detection requires a numeric column; '{column}' is not numeric")

    series = df[column]

    if method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return (series < lower) | (series > upper)

    elif method == "zscore":
        mean = series.mean()
        std = series.std()
        if std == 0:
            return pd.Series([False] * len(series), index=series.index)
        z_scores = (series - mean) / std
        return z_scores.abs() > 3

    else:
        raise ValueError(f"Unknown method '{method}'. Must be 'iqr' or 'zscore'")


def rename_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename columns using a {old_name: new_name} mapping."""
    return df.rename(columns=mapping)


def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Drop the given columns."""
    return df.drop(columns=columns)
