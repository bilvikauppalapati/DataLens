import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import decimal
import pandas as pd
import numpy as np
import pytest
from core.preprocessing import (
    handle_missing, smart_fill_missing, convert_dtype, get_compatible_dtypes,
    get_source_dtype_group, drop_duplicates, detect_outliers,
    rename_columns, drop_columns, VALID_DTYPES,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25, 30, np.nan, 40, 1000, 30],
        "name": ["a", "b", "c", None, "e", "b"],
        "price_str": ["19.99", "25.50", "10.00", "5.00", "100.00", "25.50"],
    })


# ---------- Missing values ----------

def test_handle_missing_mean(sample_df):
    result = handle_missing(sample_df, "mean", columns=["age"])
    assert result["age"].isna().sum() == 0


def test_handle_missing_invalid_strategy(sample_df):
    with pytest.raises(ValueError):
        handle_missing(sample_df, "not_a_real_strategy")


def test_handle_missing_mean_on_non_numeric_raises(sample_df):
    with pytest.raises(ValueError):
        handle_missing(sample_df, "mean", columns=["name"])


def test_smart_fill_missing_mixed_types(sample_df):
    result = smart_fill_missing(sample_df, columns=["age", "name"], numeric_strategy="mean")
    assert result["age"].isna().sum() == 0
    assert result["name"].isna().sum() == 0
    assert result["age"].iloc[2] == sample_df["age"].mean()


def test_smart_fill_missing_bad_strategy_raises(sample_df):
    with pytest.raises(ValueError):
        smart_fill_missing(sample_df, columns=["age"], numeric_strategy="ffill")


# ---------- Dtype system (str, int, float, decimal only) ----------

def test_valid_dtypes_is_exactly_four_types():
    assert VALID_DTYPES == {"str", "int", "float", "decimal"}


def test_get_source_dtype_group(sample_df):
    assert get_source_dtype_group(sample_df, "age") == "float"  # has NaN, so float
    assert get_source_dtype_group(sample_df, "name") == "str"
    assert get_source_dtype_group(sample_df, "price_str") == "str"


def test_get_source_dtype_group_int():
    df = pd.DataFrame({"i": [1, 2, 3]})
    assert get_source_dtype_group(df, "i") == "int"


def test_get_source_dtype_group_decimal():
    df = pd.DataFrame({"d": [decimal.Decimal("1.5"), decimal.Decimal("2.5")]})
    assert get_source_dtype_group(df, "d") == "decimal"


def test_get_compatible_dtypes_excludes_self():
    df = pd.DataFrame({"i": [1, 2, 3]})
    compatible = get_compatible_dtypes(df, "i")
    assert "int" not in compatible  # shouldn't offer converting to its own current type
    assert set(compatible) == {"str", "float", "decimal"}


def test_convert_dtype_str_to_float(sample_df):
    result = convert_dtype(sample_df, "price_str", "float")
    assert result["price_str"].dtype == float
    assert result["price_str"].tolist() == [19.99, 25.50, 10.00, 5.00, 100.00, 25.50]


def test_convert_dtype_round_trip_preserves_data(sample_df):
    """Regression test: str -> float -> str must NOT corrupt data
    (this is the bug we hit with bool, which this dtype system avoids)."""
    as_float = convert_dtype(sample_df, "price_str", "float")
    back_to_str = convert_dtype(as_float, "price_str", "str")
    # Values should be recognizably the same numbers, not corrupted/collapsed
    assert back_to_str["price_str"].tolist() == ["19.99", "25.5", "10.0", "5.0", "100.0", "25.5"]


def test_convert_dtype_int_to_decimal():
    df = pd.DataFrame({"qty": [1, 2, 3]})
    result = convert_dtype(df, "qty", "decimal")
    assert all(isinstance(v, decimal.Decimal) for v in result["qty"])


def test_convert_dtype_decimal_avoids_float_rounding_error():
    """Decimal conversion goes through str() first, so 0.1 stays exactly 0.1
    instead of picking up float's binary rounding artifacts."""
    df = pd.DataFrame({"val": [0.1, 0.2, 0.3]})
    result = convert_dtype(df, "val", "decimal")
    assert str(result["val"].iloc[0]) == "0.1"


def test_convert_dtype_decimal_handles_missing_values():
    df = pd.DataFrame({"val": [1.5, np.nan, 3.5]})
    result = convert_dtype(df, "val", "decimal")
    assert pd.isna(result["val"].iloc[1])
    assert result["val"].iloc[0] == decimal.Decimal("1.5")


def test_convert_dtype_bad_conversion_raises(sample_df):
    with pytest.raises(ValueError):
        convert_dtype(sample_df, "name", "int")


def test_convert_dtype_invalid_target_raises(sample_df):
    with pytest.raises(ValueError):
        convert_dtype(sample_df, "age", "bool")  # bool no longer supported


# ---------- Duplicates ----------

def test_drop_duplicates(sample_df):
    result = drop_duplicates(sample_df)
    assert len(result) == 5


# ---------- Outliers ----------

def test_detect_outliers_iqr(sample_df):
    filled = sample_df.fillna(0)
    mask = detect_outliers(filled, "age", method="iqr")
    assert mask.iloc[4] == True  # the 1000 value is an outlier


def test_detect_outliers_non_numeric_raises(sample_df):
    with pytest.raises(ValueError):
        detect_outliers(sample_df, "name", method="iqr")


# ---------- Column rename/drop ----------

def test_rename_columns(sample_df):
    result = rename_columns(sample_df, {"age": "years"})
    assert "years" in result.columns
    assert "age" not in result.columns


def test_drop_columns(sample_df):
    result = drop_columns(sample_df, ["name"])
    assert "name" not in result.columns
