import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from core.loaders import load_dataset, profile_dataset, get_extension


def test_get_extension():
    assert get_extension("sales.csv") == "csv"
    assert get_extension("data.XLSX") == "xlsx"
    assert get_extension("no_extension") == ""


def test_load_csv_basic():
    csv_bytes = b"a,b,c\n1,2,3\n4,5,6\n"
    result = load_dataset(csv_bytes, "test.csv")
    assert result.dataframe.shape == (2, 3)
    assert list(result.dataframe.columns) == ["a", "b", "c"]


def test_load_unsupported_extension():
    try:
        load_dataset(b"whatever", "file.txt")
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_profile_dataset():
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
    profile = profile_dataset(df)
    assert profile["n_rows"] == 3
    assert profile["n_cols"] == 2
    assert profile["null_counts"]["a"] == 1
