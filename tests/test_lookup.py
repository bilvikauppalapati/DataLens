import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest
from core.lookup import find_matching_rows, cross_dataset_lookup


@pytest.fixture
def countries():
    return pd.DataFrame({
        "country": ["United States", "India", "Germany", "Brazil"],
        "population": [331000000, 1380000000, 83000000, 212000000],
        "gdp_billion": [21430, 2870, 3860, 1840],
    })


@pytest.fixture
def happiness():
    return pd.DataFrame({
        "country": ["united states", " India ", "GERMANY", "Japan"],
        "happiness_score": [6.9, 4.0, 7.0, 6.1],
    })


def test_find_matching_rows_exact(countries):
    result = find_matching_rows(countries, "country", "India")
    assert result.iloc[0]["gdp_billion"] == 2870


def test_find_matching_rows_case_insensitive(happiness):
    result = find_matching_rows(happiness, "country", "india")
    assert result.iloc[0]["happiness_score"] == 4.0


def test_find_matching_rows_numeric(countries):
    result = find_matching_rows(countries, "population", 331000000)
    assert result.iloc[0]["country"] == "United States"


def test_find_matching_rows_no_match_returns_empty(countries):
    result = find_matching_rows(countries, "country", "Atlantis")
    assert result.empty


def test_find_matching_rows_unknown_column_raises(countries):
    with pytest.raises(ValueError):
        find_matching_rows(countries, "not_a_column", "x")


def test_cross_dataset_lookup_basic(countries, happiness):
    result = cross_dataset_lookup(countries, happiness, "country", "India")
    assert result.iloc[0]["happiness_score"] == 4.0


def test_cross_dataset_lookup_messy_query_value(countries, happiness):
    result = cross_dataset_lookup(countries, happiness, "country", "GERMANY")
    assert result.iloc[0]["happiness_score"] == 7.0


def test_cross_dataset_lookup_value_missing_from_second_dataset_raises(countries, happiness):
    with pytest.raises(ValueError):
        cross_dataset_lookup(countries, happiness, "country", "Brazil")


def test_cross_dataset_lookup_value_missing_from_first_dataset_raises(countries, happiness):
    with pytest.raises(ValueError):
        cross_dataset_lookup(countries, happiness, "country", "Atlantis")


def test_cross_dataset_lookup_unknown_join_column_raises(countries, happiness):
    with pytest.raises(ValueError):
        cross_dataset_lookup(countries, happiness, "not_a_column", "India")

def test_search_value_in_datasets_finds_across_columns(countries):
    from core.lookup import search_value_in_datasets
    results = search_value_in_datasets("India", {"countries.csv": countries})
    assert len(results) == 1
    assert results[0]["column"] == "country"


def test_search_value_in_datasets_numeric(countries):
    from core.lookup import search_value_in_datasets
    results = search_value_in_datasets("331000000", {"countries.csv": countries})
    assert len(results) == 1
    assert results[0]["column"] == "population"


def test_search_value_in_datasets_no_match_returns_empty_list(countries):
    from core.lookup import search_value_in_datasets
    results = search_value_in_datasets("Atlantis", {"countries.csv": countries})
    assert results == []


def test_search_value_in_datasets_multiple_datasets(countries, happiness):
    from core.lookup import search_value_in_datasets
    results = search_value_in_datasets(
        "India", {"countries.csv": countries, "happiness.csv": happiness}
    )
    dataset_names_found = {r["dataset"] for r in results}
    assert "countries.csv" in dataset_names_found
    assert "happiness.csv" in dataset_names_found