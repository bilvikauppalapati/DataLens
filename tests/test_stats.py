import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest
from core.stats import apply_filters, summary_stats, correlation_matrix, value_counts_for_column


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 1000],
        "salary": [50000, 60000, 55000, 80000, 90000, 40000],
        "country": ["US", "IN", "US", "DE", "IN", "US"],
        "name": ["a", "b", "c", "d", "e", "f"],
    })


# ---------- apply_filters ----------

def test_apply_filters_numeric_range(sample_df):
    result = apply_filters(sample_df, {"age": {"min": 30, "max": 45}})
    assert result["age"].tolist() == [30, 35, 40, 45]


def test_apply_filters_categorical(sample_df):
    result = apply_filters(sample_df, {"country": {"values": ["US", "IN"]}})
    assert set(result["country"].unique()) == {"US", "IN"}
    assert len(result) == 5


def test_apply_filters_combined(sample_df):
    result = apply_filters(sample_df, {"age": {"min": 25, "max": 40}, "country": {"values": ["US"]}})
    assert len(result) == 2
    assert all(result["country"] == "US")


def test_apply_filters_min_max_on_non_numeric_raises(sample_df):
    with pytest.raises(ValueError):
        apply_filters(sample_df, {"name": {"min": 1}})


def test_apply_filters_unknown_column_raises(sample_df):
    with pytest.raises(ValueError):
        apply_filters(sample_df, {"not_a_column": {"min": 1}})


def test_apply_filters_bad_spec_raises(sample_df):
    with pytest.raises(ValueError):
        apply_filters(sample_df, {"age": {}})


# ---------- summary_stats ----------

def test_summary_stats_numeric_column(sample_df):
    result = summary_stats(sample_df)
    assert result.loc["age", "mean"] == sample_df["age"].mean()
    assert result.loc["age", "count"] == 6


def test_summary_stats_categorical_column(sample_df):
    result = summary_stats(sample_df)
    assert result.loc["country", "unique"] == 3
    assert result.loc["country", "top"] == "US"
    assert result.loc["country", "freq"] == 3


# ---------- correlation_matrix ----------

def test_correlation_matrix_shape(sample_df):
    corr = correlation_matrix(sample_df, method="pearson")
    assert corr.shape == (2, 2)
    assert list(corr.columns) == ["age", "salary"]


def test_correlation_matrix_spearman(sample_df):
    corr = correlation_matrix(sample_df, method="spearman")
    assert corr.shape == (2, 2)


def test_correlation_matrix_invalid_method_raises(sample_df):
    with pytest.raises(ValueError):
        correlation_matrix(sample_df, method="not_a_method")


def test_correlation_matrix_insufficient_numeric_cols_raises():
    df = pd.DataFrame({"age": [1, 2, 3], "name": ["a", "b", "c"]})
    with pytest.raises(ValueError):
        correlation_matrix(df)


# ---------- value_counts_for_column ----------

def test_value_counts_for_column(sample_df):
    vc = value_counts_for_column(sample_df, "country", top_n=2)
    assert vc.iloc[0] == 3
    assert len(vc) == 2


def test_value_counts_unknown_column_raises(sample_df):
    with pytest.raises(ValueError):
        value_counts_for_column(sample_df, "not_a_column")
