import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest
from core.relations import (
    find_shared_columns_pairwise, suggest_join_keys, join_datasets, cross_dataset_correlation,
)


@pytest.fixture
def datasets():
    countries = pd.DataFrame({
        "country": ["US", "IN", "DE", "BR", "FR"],
        "population": [331000000, 1380000000, 83000000, 212000000, 67000000],
        "gdp_billion": [21430, 2870, 3860, 1840, 2780],
    })
    happiness = pd.DataFrame({
        "country": ["US", "IN", "DE", "JP"],
        "happiness_score": [6.9, 4.0, 7.0, 6.1],
        "life_expectancy": [78.9, 69.7, 81.3, 84.6],
    })
    unrelated = pd.DataFrame({"product": ["widget", "gadget"], "price": [10, 20]})
    return {"countries.csv": countries, "happiness.csv": happiness, "products.csv": unrelated}


# ---------- find_shared_columns_pairwise ----------

def test_find_shared_columns_pairwise(datasets):
    result = find_shared_columns_pairwise(datasets)
    assert len(result) == 1
    assert result.iloc[0]["shared_column"] == "country"
    assert {result.iloc[0]["dataset_a"], result.iloc[0]["dataset_b"]} == {"countries.csv", "happiness.csv"}


def test_find_shared_columns_pairwise_no_shared():
    a = pd.DataFrame({"x": [1, 2]})
    b = pd.DataFrame({"y": [1, 2]})
    result = find_shared_columns_pairwise({"a": a, "b": b})
    assert result.empty


# ---------- suggest_join_keys ----------

def test_suggest_join_keys_finds_country(datasets):
    result = suggest_join_keys(datasets, overlap_threshold=0.3)
    assert len(result) == 1
    assert result.iloc[0]["shared_column"] == "country"
    assert result.iloc[0]["common_values"] == 3


def test_suggest_join_keys_respects_threshold(datasets):
    # overlap ratio for 'country' is 3/4 = 0.75; a threshold above that should exclude it
    result = suggest_join_keys(datasets, overlap_threshold=0.9)
    assert result.empty


def test_suggest_join_keys_no_datasets_shared_returns_empty():
    a = pd.DataFrame({"x": [1, 2]})
    b = pd.DataFrame({"y": [1, 2]})
    result = suggest_join_keys({"a": a, "b": b})
    assert result.empty


# ---------- join_datasets ----------

def test_join_datasets_inner(datasets):
    merged = join_datasets(datasets["countries.csv"], datasets["happiness.csv"], "country")
    assert len(merged) == 3  # US, IN, DE match; BR/FR/JP don't


def test_join_datasets_unknown_column_raises(datasets):
    with pytest.raises(ValueError):
        join_datasets(datasets["countries.csv"], datasets["happiness.csv"], "not_a_column")


def test_join_datasets_invalid_how_raises(datasets):
    with pytest.raises(ValueError):
        join_datasets(datasets["countries.csv"], datasets["happiness.csv"], "country", how="cross_bad")


# ---------- cross_dataset_correlation ----------

def test_cross_dataset_correlation(datasets):
    corr = cross_dataset_correlation(datasets["countries.csv"], datasets["happiness.csv"], "country")
    assert "gdp_billion" in corr.columns
    assert "happiness_score" in corr.columns


def test_cross_dataset_correlation_no_overlap_raises(datasets):
    no_match = pd.DataFrame({"country": ["ZZ", "YY"], "value": [1, 2]})
    with pytest.raises(ValueError):
        cross_dataset_correlation(datasets["countries.csv"], no_match, "country")
