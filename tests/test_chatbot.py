import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.chatbot import parse_lookup_response, build_lookup_intent_prompt, build_context


DATASET_COLUMNS = {
    "countries.csv": ["country", "population", "gdp_billion"],
    "happiness.csv": ["country", "happiness_score"],
}


def test_parse_valid_single_lookup():
    raw = ('{"intent": "lookup", "scope": "single", "dataset": "countries.csv", '
           '"filter_column": "country", "filter_value": "India", "target_column": "gdp_billion"}')
    result = parse_lookup_response(raw, DATASET_COLUMNS)
    assert result["scope"] == "single"
    assert result["dataset"] == "countries.csv"


def test_parse_valid_cross_lookup():
    raw = ('{"intent": "lookup", "scope": "cross", "dataset_a": "countries.csv", '
           '"join_column": "country", "filter_value": "India", "dataset_b": "happiness.csv", '
           '"target_column": "happiness_score"}')
    result = parse_lookup_response(raw, DATASET_COLUMNS)
    assert result["scope"] == "cross"
    assert result["dataset_a"] == "countries.csv"
    assert result["dataset_b"] == "happiness.csv"


def test_parse_general_intent():
    result = parse_lookup_response('{"intent": "general"}', DATASET_COLUMNS)
    assert result == {"intent": "general"}


def test_parse_handles_markdown_fences():
    result = parse_lookup_response('```json\n{"intent": "general"}\n```', DATASET_COLUMNS)
    assert result == {"intent": "general"}


def test_parse_rejects_hallucinated_column():
    raw = ('{"intent": "lookup", "scope": "single", "dataset": "countries.csv", '
           '"filter_column": "made_up_column", "filter_value": "India", "target_column": "gdp_billion"}')
    assert parse_lookup_response(raw, DATASET_COLUMNS) is None


def test_parse_rejects_hallucinated_dataset():
    raw = ('{"intent": "lookup", "scope": "single", "dataset": "made_up.csv", '
           '"filter_column": "country", "filter_value": "India", "target_column": "gdp_billion"}')
    assert parse_lookup_response(raw, DATASET_COLUMNS) is None


def test_parse_handles_malformed_json_gracefully():
    result = parse_lookup_response("Sure! Here is the answer: India has a GDP of...", DATASET_COLUMNS)
    assert result is None


def test_parse_rejects_missing_required_fields():
    raw = ('{"intent": "lookup", "scope": "single", "dataset": "countries.csv", '
           '"filter_column": "country"}')
    assert parse_lookup_response(raw, DATASET_COLUMNS) is None


def test_parse_rejects_join_column_not_in_both_datasets():
    raw = ('{"intent": "lookup", "scope": "cross", "dataset_a": "countries.csv", '
           '"join_column": "gdp_billion", "filter_value": "5", "dataset_b": "happiness.csv", '
           '"target_column": "happiness_score"}')
    assert parse_lookup_response(raw, DATASET_COLUMNS) is None


def test_parse_rejects_non_dict_json():
    assert parse_lookup_response('["not", "a", "dict"]', DATASET_COLUMNS) is None


def test_build_lookup_intent_prompt_includes_all_datasets():
    prompt = build_lookup_intent_prompt("what is gdp for India", DATASET_COLUMNS)
    assert "countries.csv" in prompt
    assert "happiness.csv" in prompt
    assert "what is gdp for India" in prompt


def test_build_context_includes_profile():
    context = build_context(
        profile={"n_rows": 10, "n_cols": 3}, stats=None, sample_rows=[], transform_history=[]
    )
    assert "n_rows" in context