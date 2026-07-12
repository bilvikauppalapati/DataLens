import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest
from core.multi_dataset import get_common_columns, get_dataset_overview


@pytest.fixture
def datasets():
    df1 = pd.DataFrame({"country": ["US", "IN"], "population": [331000000, 1380000000], "gdp": [21430, 2870]})
    df2 = pd.DataFrame({"country": ["US", "DE"], "continent": ["NA", "EU"], "population": [331000000, 83000000]})
    df3 = pd.DataFrame({"name": ["a", "b"], "value": [1, 2]})
    return {"countries.csv": df1, "countries2.csv": df2, "unrelated.csv": df3}


def test_get_common_columns_across_all(datasets):
    assert get_common_columns(datasets) == []


def test_get_common_columns_across_subset(datasets):
    subset = {"a": datasets["countries.csv"], "b": datasets["countries2.csv"]}
    assert get_common_columns(subset) == ["country", "population"]


def test_get_common_columns_empty_input():
    assert get_common_columns({}) == []


def test_get_dataset_overview_shape(datasets):
    overview = get_dataset_overview(datasets)
    assert overview.loc["countries.csv", "rows"] == 2
    assert overview.loc["countries.csv", "columns"] == 3
    assert overview.loc["unrelated.csv", "columns"] == 2


def test_get_dataset_overview_duplicate_count():
    df_with_dupes = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    overview = get_dataset_overview({"dupes.csv": df_with_dupes})
    assert overview.loc["dupes.csv", "duplicate_rows"] == 1
