import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from core.sql_query import (
    is_safe_select, clean_sql_response, build_sql_prompt, build_table_aliases,
    run_sql, run_sql_multi,
)


@pytest.fixture
def cars():
    return pd.DataFrame({
        "Car_ID": [1, 2, 3, 4, 5],
        "Brand": ["Ford", "BMW", "BMW", "Tesla", "BMW"],
        "Model_Year": [2023, 2018, 2023, 2021, 2023],
        "Fuel_Type": ["Petrol", "Diesel", "Diesel", "Electric", "Petrol"],
        "Price": [30000.0, 45000.0, 60000.0, 35000.0, 48000.0],
    })


def test_is_safe_select_accepts_plain_select():
    assert is_safe_select("SELECT * FROM cars WHERE Brand = 'BMW'")


def test_is_safe_select_accepts_with_cte():
    assert is_safe_select("WITH t AS (SELECT * FROM cars) SELECT * FROM t")


def test_is_safe_select_case_insensitive():
    assert is_safe_select("select brand from cars")


def test_is_safe_select_rejects_empty():
    assert not is_safe_select("")
    assert not is_safe_select("   ")


def test_is_safe_select_rejects_non_select_start():
    assert not is_safe_select("UPDATE cars SET Price = 0")
    assert not is_safe_select("DELETE FROM cars")


def test_is_safe_select_rejects_forbidden_keyword_anywhere():
    assert not is_safe_select("SELECT * FROM cars; DROP TABLE cars")
    assert not is_safe_select("SELECT * FROM cars WHERE Brand = (INSERT INTO x VALUES (1))")


def test_is_safe_select_rejects_multiple_statements():
    assert not is_safe_select("SELECT * FROM cars; SELECT * FROM cars")


def test_clean_sql_response_strips_markdown_fence():
    result = clean_sql_response("```sql\nSELECT * FROM cars\n```")
    assert result == "SELECT * FROM cars"


def test_clean_sql_response_strips_plain_fence():
    result = clean_sql_response("```\nSELECT * FROM cars\n```")
    assert result == "SELECT * FROM cars"


def test_clean_sql_response_strips_trailing_semicolon():
    result = clean_sql_response("SELECT * FROM cars;")
    assert result == "SELECT * FROM cars"


def test_clean_sql_response_fixes_redundant_backtick_around_quoted_ident():
    """Regression test for a real failure seen with the local model: it
    wrapped an already double-quoted table name in backticks too
    (`` `"cars.csv"` ``), which SQLite parses as a table literally named
    `"cars.csv"` (quote characters included) — a table that doesn't exist
    — so an otherwise-correct query failed with 'no such table'."""
    result = clean_sql_response('SELECT * FROM `"cars.csv"` WHERE Brand = \'BMW\'')
    assert result == 'SELECT * FROM "cars.csv" WHERE Brand = \'BMW\''


def test_run_sql_multi_handles_redundant_backtick_quoting():
    df = pd.DataFrame({"Brand": ["Ford", "BMW"]})
    result = run_sql_multi('SELECT * FROM `"cars.csv"` WHERE Brand = \'BMW\'', {"cars.csv": df})
    assert len(result) == 1


def test_build_table_aliases_strips_extension_and_dots():
    """Regression test for a real failure: the model was unreliable about
    quoting a dotted filename like 'car_price_dataset.csv' as a SQL table
    name — sometimes dropping the quotes (SQLite then misparses the dot as
    schema.table) and sometimes dropping the extension entirely. Aliasing
    to a bare, dot-free identifier sidesteps the whole failure class."""
    aliases = build_table_aliases({"car_price_dataset.csv": pd.DataFrame()})
    assert aliases == {"car_price_dataset.csv": "car_price_dataset"}


def test_build_table_aliases_handles_collisions():
    aliases = build_table_aliases({
        "cars.csv": pd.DataFrame(), "cars.xlsx": pd.DataFrame(), "cars!.json": pd.DataFrame(),
    })
    assert len(set(aliases.values())) == 3
    for alias in aliases.values():
        assert re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", alias)


def test_build_sql_prompt_uses_alias_not_raw_dotted_name(cars):
    prompt = build_sql_prompt("how many BMW cars", {"car_price_dataset.csv": cars})
    assert "Table car_price_dataset " in prompt
    assert "car_price_dataset.csv" in prompt  # still mentioned for context


def test_build_sql_prompt_includes_table_and_columns(cars):
    prompt = build_sql_prompt("how many BMW cars are there", {"cars.csv": cars})
    assert "cars.csv" in prompt
    assert "Brand" in prompt
    assert "Price" in prompt
    assert "how many BMW cars are there" in prompt


def test_build_sql_prompt_includes_every_table(cars):
    happiness = pd.DataFrame({"Brand": ["Ford", "BMW"], "Rating": [4, 5]})
    prompt = build_sql_prompt("join these", {"cars.csv": cars, "ratings.csv": happiness})
    assert "cars.csv" in prompt
    assert "ratings.csv" in prompt
    assert "Rating" in prompt


def test_run_sql_executes_real_select(cars):
    result = run_sql("SELECT COUNT(*) AS n FROM cars WHERE Brand = 'BMW'", "cars", cars)
    assert result["n"].iloc[0] == 3


def test_run_sql_executes_aggregate_sum(cars):
    """Aggregation is exactly what the rule-based nlq.py detectors can't
    do at all — this is the core capability this module adds."""
    result = run_sql("SELECT SUM(Price) AS total FROM cars WHERE Brand = 'BMW'", "cars", cars)
    assert result["total"].iloc[0] == pytest.approx(45000.0 + 60000.0 + 48000.0)


def test_run_sql_executes_distinct(cars):
    result = run_sql("SELECT DISTINCT Brand FROM cars", "cars", cars)
    assert set(result["Brand"]) == {"Ford", "BMW", "Tesla"}


def test_run_sql_executes_group_by(cars):
    result = run_sql(
        "SELECT Brand, COUNT(*) AS n FROM cars GROUP BY Brand ORDER BY Brand", "cars", cars
    )
    assert dict(zip(result["Brand"], result["n"])) == {"Ford": 1, "BMW": 3, "Tesla": 1}


def test_run_sql_rejects_unsafe_sql(cars):
    with pytest.raises(ValueError):
        run_sql("DELETE FROM cars", "cars", cars)


def test_run_sql_raises_valueerror_on_bad_column(cars):
    with pytest.raises(ValueError):
        run_sql("SELECT made_up_column FROM cars", "cars", cars)


def test_run_sql_does_not_mutate_original_dataframe(cars):
    original = cars.copy()
    run_sql("SELECT * FROM cars WHERE Brand = 'BMW'", "cars", cars)
    pd.testing.assert_frame_equal(cars, original)


def test_run_sql_multi_joins_across_tables(cars):
    """This is what supersedes the old hardcoded 'cross-dataset lookup'
    JSON schema: a real SQL JOIN across two tables in one query."""
    ratings = pd.DataFrame({"Brand": ["Ford", "BMW", "Tesla"], "Rating": [3, 5, 4]})
    result = run_sql_multi(
        'SELECT c."Brand", r."Rating" FROM "cars.csv" c '
        'JOIN "ratings.csv" r ON c."Brand" = r."Brand" '
        'WHERE c."Brand" = \'BMW\' LIMIT 1',
        {"cars.csv": cars, "ratings.csv": ratings},
    )
    assert result["Rating"].iloc[0] == 5


def test_run_sql_multi_handles_dotted_table_names(cars):
    result = run_sql_multi('SELECT COUNT(*) AS n FROM "cars.csv"', {"cars.csv": cars})
    assert result["n"].iloc[0] == 5
