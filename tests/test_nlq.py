import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest
from core.nlq import extract_candidate_values, guess_target_column, detect_lookup_rule_based


@pytest.fixture
def cars():
    return pd.DataFrame({
        "Car_ID": [1, 2, 12, 13],
        "Brand": ["Ford", "Hyundai", "BMW", "Tesla"],
        "Horsepower": [82, 259, 325, 111],
        "Price": [34309.25, 55153.6, 49052.9, 34220.1],
    })


@pytest.fixture
def datasets(cars):
    return {"cars.csv": cars}


def test_extract_numbers():
    result = extract_candidate_values("what is brand for car id 12")
    assert "12" in result


def test_extract_quoted_phrase():
    result = extract_candidate_values('what is the score for "New Delhi"')
    assert "New Delhi" in result


def test_extract_capitalized_word():
    result = extract_candidate_values("what is the happiness score for India")
    assert "India" in result


def test_extract_skips_stopwords():
    result = extract_candidate_values("What is the Brand")
    assert "What" not in result


def test_guess_target_column_basic():
    result = guess_target_column("what is brand name with car id 12", ["Car_ID", "Brand", "Price"])
    assert result in ("Car_ID", "Brand")


def test_guess_target_column_excludes_filter_column():
    """Regression test for the exact bug reported: asking to filter BY Car_ID
    and also look up Car_ID itself isn't useful — Brand should win once
    Car_ID is excluded as the already-used filter column."""
    result = guess_target_column(
        "what is brand name with car id 12", ["Car_ID", "Brand", "Price"], exclude=["Car_ID"]
    )
    assert result == "Brand"


def test_guess_target_column_matches_literal_underscore_form():
    """Regression test: a user typing the literal column header
    ('model_year') must match just as well as natural phrasing
    ('model year') — previously only the column side was normalized."""
    result = guess_target_column("list all model_year values", ["Car_ID", "Brand", "Model_Year"])
    assert result == "Model_Year"


def test_guess_target_column_no_match_returns_none():
    result = guess_target_column("tell me something random", ["Car_ID", "Brand"])
    assert result is None


def test_detect_lookup_rule_based_exact_reported_bug(datasets):
    """The exact question that failed in production: chatbot said 'not
    enough information' even though Car_ID 12 / Brand=BMW was in the data."""
    result = detect_lookup_rule_based("what is brand name with car id 12", datasets)
    assert result is not None
    assert result["dataset"] == "cars.csv"
    assert result["filter_column"] == "Car_ID"
    assert result["filter_value"] == "12"
    assert result["target_column"] == "Brand"


def test_detect_lookup_rule_based_horsepower_variant(datasets):
    result = detect_lookup_rule_based("show me the horsepower of car id 12", datasets)
    assert result["target_column"] == "Horsepower"


def test_detect_lookup_rule_based_nonexistent_id_returns_none(datasets):
    result = detect_lookup_rule_based("what is brand for car id 99999", datasets)
    assert result is None


def test_detect_lookup_rule_based_declines_non_unique_match(cars_full):
    """Regression test: 'what is the price for BMW' where Brand=BMW matches
    MULTIPLE rows with DIFFERENT prices must not silently report the first
    matching row's price as 'the' answer — that's a real wrong answer
    (e.g. for 'total price of BMW cars'), not just an imprecise one. Must
    defer so the caller can fall back to a real aggregate/filter instead."""
    result = detect_lookup_rule_based("what is the price for BMW", {"cars.csv": cars_full})
    assert result is None


def test_detect_lookup_rule_based_accepts_unique_target_across_matches():
    """Same shape (filter value matches multiple rows), but the target
    column happens to be IDENTICAL across all of them — still a valid,
    non-misleading single-value answer."""
    df = pd.DataFrame({
        "Car_ID": [1, 2, 3],
        "Brand": ["BMW", "BMW", "Tesla"],
        "Country": ["Germany", "Germany", "USA"],
    })
    result = detect_lookup_rule_based("what is the country for BMW", {"cars.csv": df})
    assert result is not None
    assert result["target_column"] == "Country"


def test_detect_lookup_rule_based_general_question_returns_none(datasets):
    result = detect_lookup_rule_based("summarize this dataset for me", datasets)
    assert result is None


def test_detect_lookup_rule_based_no_target_column_returns_none(datasets):
    result = detect_lookup_rule_based("12", datasets)
    assert result is None


@pytest.fixture
def cars_full():
    return pd.DataFrame({
        "Car_ID": [1, 2, 3, 4, 5],
        "Brand": ["Ford", "BMW", "BMW", "Tesla", "BMW"],
        "Doors": [2, 4, 2, 4, 4],
        "Price": [30000.0, 45000.0, 60000.0, 35000.0, 48000.0],
        "Fuel_Type": ["Petrol", "Petrol", "Diesel", "Electric", "Petrol"],
        "Horsepower": [200, 320, 280, 350, 310],
    })


def test_detect_filter_query_between(cars_full):
    from core.nlq import detect_filter_query
    result = detect_filter_query("list cars with price between 30000 and 50000", cars_full)
    assert result is not None
    assert result["filters"]["Price"] == {"min": 30000.0, "max": 50000.0}


def test_detect_filter_query_threshold_and_equality(cars_full):
    from core.nlq import detect_filter_query
    result = detect_filter_query("show cars with horsepower above 300 and fuel type Petrol", cars_full)
    assert result is not None
    assert result["filters"]["Horsepower"] == {"min": 300.0}
    assert result["filters"]["Fuel_Type"] == {"values": ["Petrol"]}


def test_detect_filter_query_two_equality_conditions_uses_correct_columns(cars_full):
    """Regression test: 'doors 4' must match the Doors column, not accidentally
    match Car_ID just because a Car_ID of 4 also exists in the data."""
    from core.nlq import detect_filter_query
    result = detect_filter_query("cars with brand BMW and doors 4", cars_full)
    assert result is not None
    assert result["filters"]["Brand"] == {"values": ["BMW"]}
    assert result["filters"]["Doors"] == {"values": [4]}
    assert "Car_ID" not in result["filters"]


def test_detect_filter_query_single_equality_returns_none(cars_full):
    from core.nlq import detect_filter_query
    result = detect_filter_query("what is brand for car id 4", cars_full)
    assert result is None


def test_detect_filter_query_single_threshold(cars_full):
    from core.nlq import detect_filter_query
    result = detect_filter_query("cars priced below 40000", cars_full)
    assert result is not None
    assert result["filters"]["Price"] == {"max": 40000.0}


def test_detect_filter_query_implicit_multi_condition_no_connector(cars_full):
    """Regression test for the exact reported bug: two conditions ('brand
    BMW' and 'model_year 2023') expressed with NO 'and'/',' connector must
    still both be captured, not silently dropped down to one and then
    rejected entirely (which was falling through to a wrong single-row
    answer that ignored the brand filter)."""
    from core.nlq import detect_filter_query
    cars_full = cars_full.assign(Model_Year=[2023, 2018, 2023, 2021, 2023])
    result = detect_filter_query("i want car brand BMW model_year of 2023", cars_full)
    assert result is not None
    assert result["filters"]["Brand"] == {"values": ["BMW"]}
    assert result["filters"]["Model_Year"] == {"values": [2023]}


def test_detect_filter_query_implicit_multi_condition_lowercase_brand(cars_full):
    """Same bug, but the brand is typed lowercase ('bmw') with no explicit
    'brand' column name mentioned either — must resolve via real data
    values, not capitalization or named-column heuristics."""
    from core.nlq import detect_filter_query
    cars_full = cars_full.assign(Model_Year=[2023, 2018, 2023, 2021, 2023])
    result = detect_filter_query("list out bmw car models in year 2023", cars_full)
    assert result is not None
    assert result["filters"]["Brand"] == {"values": ["BMW"]}
    assert result["filters"]["Model_Year"] == {"values": [2023]}


def test_detect_filter_query_multiple_values_same_column_via_and():
    """Regression test for the exact reported bug: 'bmw and tesla' joins
    TWO values of the SAME column (Brand) with 'and', not two conditions
    on different columns. Clause-splitting on 'and' then hands each half
    ('bmw' / 'tesla') to step 2 with no capitalized/named candidate to
    latch onto, silently dropping both. Must resolve to an OR-style
    multi-value Brand filter, combined with the separate Model_Year
    condition, instead of falling through to a wrong single-row guess."""
    from core.nlq import detect_filter_query
    df = pd.DataFrame({
        "Car_ID": [1, 2, 3, 4, 5, 6],
        "Brand": ["Ford", "Hyundai", "BMW", "Tesla", "Tesla", "BMW"],
        "Model_Year": [2023, 2018, 2023, 2023, 2013, 2009],
    })
    result = detect_filter_query("list out bmw and tesla car models in year 2023", df)
    assert result is not None
    assert set(result["filters"]["Brand"]["values"]) == {"BMW", "Tesla"}
    assert result["filters"]["Model_Year"] == {"values": [2023]}


def test_detect_filter_query_no_filter_content_returns_none(cars_full):
    from core.nlq import detect_filter_query
    result = detect_filter_query("summarize this dataset", cars_full)
    assert result is None


def test_detect_filter_query_hyphen_range(cars_full):
    """Regression test: 'between 2020-2023' (hyphen instead of 'and'/'to')
    must be recognized as a range, not silently ignored."""
    from core.nlq import detect_filter_query
    cars_full = cars_full.assign(Year=[2018, 2020, 2021, 2023, 2025])
    result = detect_filter_query("cars made in year between 2020-2023", cars_full)
    assert result is not None
    assert result["filters"]["Year"] == {"min": 2020.0, "max": 2023.0}


def test_detect_filter_query_bare_to_range_no_between_keyword(cars_full):
    """Regression test: 'model_year 2020 to 2023' — no 'between' keyword
    at all — must still be recognized as a range. Previously the range
    regex only fired on literal 'between', so this phrasing went
    undetected entirely and (worse) let 'list' hijack it into the
    distinct-values path, dumping every year instead of filtering."""
    from core.nlq import detect_filter_query
    cars_full = cars_full.assign(Model_Year=[2018, 2020, 2021, 2023, 2025])
    result = detect_filter_query("model_year 2020 to 2023", cars_full)
    assert result is not None
    assert result["filters"]["Model_Year"] == {"min": 2020.0, "max": 2023.0}


def test_detect_distinct_values_query_defers_on_bare_to_range(cars_full):
    """Same regression, from the distinct-values side: the 'list' trigger
    must not hijack a bare 'X to Y' range question."""
    from core.nlq import detect_distinct_values_query
    cars_full = cars_full.assign(Model_Year=[2018, 2020, 2021, 2023, 2025])
    result = detect_distinct_values_query(
        "list out cars from model_year 2020 to 2023 from bmw and tesla", cars_full
    )
    assert result is None


def test_detect_filter_query_hyphen_range_no_spaces(cars_full):
    from core.nlq import detect_filter_query
    cars_full = cars_full.assign(Year=[2018, 2020, 2021, 2023, 2025])
    result = detect_filter_query("year between 2020 - 2023", cars_full)
    assert result is not None
    assert result["filters"]["Year"] == {"min": 2020.0, "max": 2023.0}


def test_detect_distinct_values_query_reported_bug(cars_full):
    """Regression test for the exact bug reported: 'total 6 car brand names'
    must list the real distinct Brand values, not mistake '6' for a Car_ID
    lookup and return one wrong row."""
    from core.nlq import detect_distinct_values_query
    result = detect_distinct_values_query("total 6 car brand names", cars_full)
    assert result is not None
    assert result["column"] == "Brand"
    assert set(result["values"]) == {"Ford", "BMW", "Tesla"}


def test_detect_distinct_values_query_list_all_phrasing(cars_full):
    from core.nlq import detect_distinct_values_query
    result = detect_distinct_values_query("list all fuel types", cars_full)
    assert result is not None
    assert result["column"] == "Fuel_Type"
    assert set(result["values"]) == {"Petrol", "Diesel", "Electric"}


def test_detect_distinct_values_query_unique_phrasing(cars_full):
    from core.nlq import detect_distinct_values_query
    result = detect_distinct_values_query("how many unique doors values are there", cars_full)
    assert result is not None
    assert result["column"] == "Doors"
    assert set(result["values"]) == {2, 4}


def test_detect_distinct_values_query_defers_to_id_lookup(cars_full):
    """Must not steal id-based lookups even though 'id' questions can
    otherwise contain trigger-ish words."""
    from core.nlq import detect_distinct_values_query
    result = detect_distinct_values_query("what is brand for car id 4", cars_full)
    assert result is None


def test_detect_distinct_values_query_defers_to_threshold_query(cars_full):
    from core.nlq import detect_distinct_values_query
    result = detect_distinct_values_query("how many cars have horsepower above 300", cars_full)
    assert result is None


def test_detect_distinct_values_query_no_trigger_returns_none(cars_full):
    from core.nlq import detect_distinct_values_query
    result = detect_distinct_values_query("what is the price for BMW", cars_full)
    assert result is None


def test_detect_distinct_values_query_defers_on_bare_all(cars_full):
    """Regression test for a real-world bug: 'in which year did all car
    brands have their model released' was hijacked by the bare word 'all'
    and answered with a wrong, confident 'distinct Brand values' response
    instead of admitting this cross-brand question isn't something the
    rule-based detectors support. 'all' alone must not be a trigger —
    'list'/'distinct'/'unique'/etc. still are."""
    from core.nlq import detect_distinct_values_query
    result = detect_distinct_values_query(
        "in which year did all car brands have their model released", cars_full
    )
    assert result is None


def test_detect_distinct_values_query_defers_on_ambiguous_total_numeric(cars_full):
    """Regression test: 'total'/'how many' are ambiguous between 'list
    distinct values' and 'sum this column' (which this app can't compute).
    On a NUMERIC column, must defer (None) rather than dump a meaningless
    list of every distinct number."""
    from core.nlq import detect_distinct_values_query
    result = detect_distinct_values_query("what's the total horsepower of all BMW cars", cars_full)
    assert result is None


def test_detect_distinct_values_query_total_still_works_on_categorical(cars_full):
    """Same ambiguous word 'total', but on a categorical column there's no
    sane 'sum' reading, so it should still resolve to the distinct list."""
    from core.nlq import detect_distinct_values_query
    result = detect_distinct_values_query("total car brand names", cars_full)
    assert result is not None
    assert result["column"] == "Brand"


def test_detect_distinct_values_query_list_out_all_phrasing():
    """Regression test for a real-world bug: 'list out all' (word 'out' in
    between) must still trigger, not just the exact phrase 'list all'.
    Uses a dataset where only 3 of the 6 real brands appear in the first 5
    rows, to catch the failure mode of quietly falling back to the AI path
    (which only ever sees df.head(5)) instead of scanning the real column."""
    from core.nlq import detect_distinct_values_query
    cars = pd.DataFrame({
        "Car_ID": list(range(1, 11)),
        "Brand": ["Ford", "Hyundai", "BMW", "Hyundai", "Hyundai",
                  "Honda", "Tesla", "Toyota", "Ford", "BMW"],
    })
    result = detect_distinct_values_query("list out all avaiable car brands in dataset", cars)
    assert result is not None
    assert result["column"] == "Brand"
    assert set(result["values"]) == {"Ford", "Hyundai", "BMW", "Honda", "Tesla", "Toyota"}


def test_resolve_structured_query_filter_wins_over_distinct_trigger(cars_full):
    """Regression test for the exact reported bug: 'list out bmw and
    tesla with 4 doors' has a real, resolvable filter condition (Brand IN
    (BMW, Tesla) AND Doors = 4), but also contains the word 'list' and a
    real column name ('doors') on its own. Calling the two detectors
    directly in the wrong order let the generic 'list' trigger hijack it
    into 'show every distinct Doors value', discarding the actual
    conditions. resolve_structured_query is the one place this priority
    is enforced, so it must return the filter, not the distinct dump."""
    from core.nlq import resolve_structured_query
    result = resolve_structured_query("list out bmw and tesla with 4 doors", cars_full)
    assert result is not None
    assert result["kind"] == "filter"
    assert set(result["filters"]["Brand"]["values"]) == {"BMW", "Tesla"}
    assert result["filters"]["Doors"] == {"values": [4]}


def test_resolve_structured_query_range_wins_over_distinct_trigger(cars_full):
    from core.nlq import resolve_structured_query
    cars_full = cars_full.assign(Model_Year=[2018, 2020, 2021, 2023, 2025])
    result = resolve_structured_query(
        "list out cars from model_year 2020 to 2023 from bmw and tesla", cars_full
    )
    assert result["kind"] == "filter"
    assert result["filters"]["Model_Year"] == {"min": 2020.0, "max": 2023.0}
    assert set(result["filters"]["Brand"]["values"]) == {"BMW", "Tesla"}


def test_resolve_structured_query_falls_back_to_distinct(cars_full):
    """When there's genuinely no filter condition, the distinct-values
    detector must still be reachable — the reordering must not starve it."""
    from core.nlq import resolve_structured_query
    result = resolve_structured_query("list all fuel types", cars_full)
    assert result is not None
    assert result["kind"] == "distinct"
    assert result["column"] == "Fuel_Type"


def test_resolve_structured_query_returns_none_when_neither_matches(cars_full):
    from core.nlq import resolve_structured_query
    result = resolve_structured_query("summarize this dataset", cars_full)
    assert result is None


def test_detect_distinct_values_query_defers_when_scoped_by_other_column(cars_full):
    """Regression test for the exact reported bug: 'list all model_year of
    Honda' has a real column trigger (Model_Year) but is actually SCOPED
    to another column's value ('Honda'). Returning every Model_Year in
    the whole dataset — ignoring 'of Honda' — is a wrong, misleading
    answer, not just an imprecise one. Must defer so it can fall through
    to a real SQL query, which can express 'DISTINCT x WHERE brand = y'."""
    from core.nlq import detect_distinct_values_query
    result = detect_distinct_values_query("list all model_year of Honda", cars_full)
    assert result is None


def test_detect_distinct_values_query_unscoped_still_works(cars_full):
    """Same target column, but genuinely unscoped (no other column's real
    value mentioned) — must still resolve normally."""
    from core.nlq import detect_distinct_values_query
    cars_with_year = cars_full.assign(Model_Year=[2020, 2021, 2022, 2023, 2019])
    result = detect_distinct_values_query("list all model_year values", cars_with_year)
    assert result is not None
    assert result["column"] == "Model_Year"