"""
nlq.py — Phase 8 support module
Rule-based ("deterministic") natural-language lookup detection.

This exists because relying on a small local LLM to correctly format a
JSON description of every question is not reliable enough on its own —
it silently fails sometimes, which then falls back to a general answer
that can only see a few sample rows (see core.chatbot.detect_lookup_intent).

This module instead tries to answer the same question with plain code:
  1. Pull out candidate "values" mentioned in the question (numbers,
     quoted text, capitalized words).
  2. Actually search the real data for each candidate value, across every
     column of every loaded dataset (core.lookup.search_value_in_datasets)
     — a real search, not a guess, so it works no matter how large the
     dataset is.
  3. Guess which column the user wants back, by matching words in the
     question against real column names.

If this can't confidently resolve the question, it returns None so the
caller can fall back to the LLM-based intent detection instead.
"""
from __future__ import annotations

from core.lookup import search_value_in_datasets, find_matching_rows

import re

import pandas as pd

from core.lookup import search_value_in_datasets


# Common words that show up in questions but are never actual data values.
_STOPWORDS = {
    "what", "whats", "where", "which", "who", "how", "is", "are", "was",
    "were", "the", "a", "an", "with", "for", "of", "in", "on", "based",
    "id", "name", "value", "does", "do", "has", "have", "tell", "me",
    "show", "find", "give", "and", "or", "to", "from", "that", "this",
}


def extract_candidate_values(question: str) -> list[str]:
    """
    Pulls out plausible "values" from a question: quoted phrases first
    (most explicit), then standalone numbers, then capitalized words
    (likely proper nouns like a country or brand name), skipping common
    question words. Order matters — more specific candidates come first.
    """
    candidates: list[str] = []

    # Quoted phrases: "BMW" or 'BMW'
    for match in re.findall(r'"([^"]+)"|\'([^\']+)\'', question):
        value = match[0] or match[1]
        if value:
            candidates.append(value)

    # Standalone numbers (e.g. an ID like 12)
    for match in re.findall(r"\b\d+\b", question):
        candidates.append(match)

    # Capitalized words that aren't common question words (e.g. India, BMW)
    for match in re.findall(r"\b[A-Z][A-Za-z]*\b", question):
        if match.lower() not in _STOPWORDS:
            candidates.append(match)

    # De-duplicate while preserving order
    seen = set()
    unique = []
    for c in candidates:
        if c.lower() not in seen:
            seen.add(c.lower())
            unique.append(c)
    return unique


def guess_target_column(question: str, columns: list[str], exclude: list[str] | None = None) -> str | None:
    """
    Finds which real column name the question is most likely asking about,
    by checking whether the column's name (with underscores treated as
    spaces) appears as words in the question. Returns the longest/most
    specific match, or None if nothing lines up.

    `exclude` (e.g. the column already used as the filter) is skipped —
    asking to filter BY a column and asking to look THAT SAME column up
    again isn't a useful target, so it doesn't count as a valid guess.
    """
    # Normalize underscores/hyphens on BOTH sides — a user who types the
    # literal column header ("model_year") should match just as well as
    # one who writes it naturally ("model year").
    q_normalized = question.lower().replace("_", " ").replace("-", " ")
    exclude_set = {c.lower() for c in (exclude or [])}
    best_match = None
    best_len = 0

    for col in columns:
        if col.lower() in exclude_set:
            continue
        col_words = col.lower().replace("_", " ").replace("-", " ")
        if col_words in q_normalized and len(col_words) > best_len:
            best_match = col
            best_len = len(col_words)

    return best_match


def detect_lookup_rule_based(question: str, datasets: dict[str, pd.DataFrame]) -> dict | None:
    """
    Tries to resolve the question into a lookup using plain code — no LLM.
    Returns a dict shaped like:
        {"intent": "lookup", "scope": "single", "dataset": ..., "filter_column": ...,
         "filter_value": ..., "target_column": ...}
    or None if it can't confidently figure out both the filter value's
    location AND the target column being asked for.
    """
    candidates = extract_candidate_values(question)

    for value in candidates:
        matches = search_value_in_datasets(value, datasets)
        if not matches:
            continue

        # For each dataset/column where this value was actually found,
        # see if we can also confidently guess which column is being asked about.
        resolved = []
        for m in matches:
            dataset_name = m["dataset"]
            target_col = guess_target_column(
                question, list(datasets[dataset_name].columns), exclude=[m["column"]]
            )
            if not target_col:
                continue
            # The filter value must pin down a SINGLE answer — if it matched
            # many rows (e.g. "BMW" matches every BMW, not just one car) and
            # those rows disagree on the target column, reporting the first
            # one as "the" answer is just wrong, not merely imprecise (e.g.
            # "what is the total price of BMW cars" silently returning one
            # arbitrary car's price). Only accept the match when the target
            # column is actually consistent across every matching row —
            # otherwise defer to the filter/SQL paths, which return the
            # real set of rows (or a real aggregate) instead of one guess.
            if m["rows"][target_col].nunique(dropna=False) > 1:
                continue
            resolved.append({
                "intent": "lookup", "scope": "single", "dataset": dataset_name,
                "filter_column": m["column"], "filter_value": value,
                "target_column": target_col,
            })

        # Only act if exactly one (dataset, target_column) resolution makes
        # sense — if it's ambiguous, don't guess, fall back instead.
        unique_resolutions = {(r["dataset"], r["target_column"]) for r in resolved}
        if len(unique_resolutions) == 1:
            return resolved[0]

    return None

# ---------------------------------------------------------------------------
# Range and multi-condition filter queries
#
# The single-value lookup above assumes ONE filter condition and ONE
# target column ("what is X where Y = Z"). It has no concept of ranges
# ("between A and B") or combining multiple conditions ("X and Y") — and
# worse, it would previously misfire on these by picking just one
# condition and ignoring the rest. This section detects those queries
# properly and returns a filter spec compatible with core.stats.apply_filters.
# ---------------------------------------------------------------------------

_RANGE_PATTERN = re.compile(
    r"([a-zA-Z_ ]*?)\b(?:between\s+)?([\d.]+)(?:\s+(?:and|to)\s+|\s*-\s*)([\d.]+)", re.IGNORECASE
)
_MIN_PATTERN = re.compile(
    r"(above|over|greater than|more than|at least)\s+([\d.]+)", re.IGNORECASE
)
_MAX_PATTERN = re.compile(
    r"(below|under|less than|at most)\s+([\d.]+)", re.IGNORECASE
)


def detect_filter_query(question: str, df: pd.DataFrame) -> dict | None:
    """
    Detects range ("between X and Y") and/or multiple combined conditions
    ("X above 300 and Y is Petrol") in a question, scoped to a single
    dataframe. Returns {"filters": {...}} in the same shape
    core.stats.apply_filters expects, or None if no such condition is found.

    This intentionally only returns a result when there's a RANGE/THRESHOLD
    condition, or MORE THAN ONE condition — a single plain equality ("car id
    12") is left for the simpler, more precise single-value lookup instead.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = list(df.columns)
    filters: dict = {}

    working_text = question

    # 1. "between A and B" — extract first, since it contains the word
    # "and" which would otherwise confuse clause-splitting below.
    for match in _RANGE_PATTERN.finditer(working_text):
        prefix, low_str, high_str = match.groups()
        low, high = float(low_str), float(high_str)
        if low > high:
            low, high = high, low
        col = guess_target_column(prefix, numeric_cols) or guess_target_column(question, numeric_cols)
        if col:
            filters[col] = {"min": low, "max": high}

    working_text = _RANGE_PATTERN.sub(" ", working_text)

    # 2. Split what's left into clauses on "and" / "," for the remaining conditions
    clauses = re.split(r"\band\b|,", working_text, flags=re.IGNORECASE)

    # Values that successfully resolved to a filter below — tracked so step 3
    # (which re-scans the same text when there was no "and"/",") doesn't
    # reprocess a value that's already spoken for, which would otherwise let
    # it accidentally latch onto a second, unrelated column (e.g. "4" already
    # matched to Car_ID also happens to equal a Doors value).
    used_candidate_values: set[str] = set()

    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue

        min_match = _MIN_PATTERN.search(clause)
        max_match = _MAX_PATTERN.search(clause)

        if min_match:
            col = guess_target_column(clause, numeric_cols)
            if col:
                filters.setdefault(col, {})["min"] = float(min_match.group(2))
                continue

        if max_match:
            col = guess_target_column(clause, numeric_cols)
            if col:
                filters.setdefault(col, {})["max"] = float(max_match.group(2))
                continue

        # Otherwise: try to find an equality condition (column = value) in this clause.
        # Prefer whichever column is EXPLICITLY NAMED in the clause (e.g. "doors 4"
        # names "Doors") over a blind search across every column — a blind search
        # can accidentally match the value in an unrelated column (e.g. an ID).
        named_col = guess_target_column(clause, all_cols)
        candidates = extract_candidate_values(clause)

        matched_col = None
        actual_value = None
        matched_value = None

        if named_col:
            for value in candidates:
                try:
                    matches = find_matching_rows(df, named_col, value)
                except ValueError:
                    matches = pd.DataFrame()
                if not matches.empty:
                    matched_col = named_col
                    actual_value = matches[named_col].iloc[0]
                    matched_value = value
                    break

        if matched_col is None:
            # No column explicitly named (or it didn't match) — fall back to
            # searching every column for the value, as a last resort.
            for value in candidates:
                for col in all_cols:
                    try:
                        matches = find_matching_rows(df, col, value)
                    except ValueError:
                        continue
                    if not matches.empty:
                        matched_col = col
                        actual_value = matches[col].iloc[0]
                        matched_value = value
                        break
                if matched_col:
                    break

        if matched_col:
            filters.setdefault(matched_col, {}).setdefault("values", [])
            if actual_value not in filters[matched_col]["values"]:
                filters[matched_col]["values"].append(actual_value)
            used_candidate_values.add(matched_value)

    # 3a. Categorical multi-value scan — ALWAYS runs across the whole
    # remaining text, regardless of clause count. "and" doesn't only
    # separate DIFFERENT conditions ("brand BMW and doors 4") — it's just
    # as often used to list MULTIPLE values of the SAME column ("bmw and
    # tesla car models"), and clause-splitting then hands step 2 two
    # half-sentences, each with no capitalized/quoted/numeric candidate
    # (a lowercase, unnamed "bmw" / "tesla" on its own), so both values get
    # silently dropped. This matches real column values directly against
    # the text instead — case-insensitively, and collecting EVERY matching
    # value per column (a genuine "OR" filter), not just the first.
    for col in all_cols:
        if col in numeric_cols:
            continue
        spec = filters.get(col, {})
        if "min" in spec or "max" in spec:
            continue
        try:
            real_values = df[col].dropna().unique().tolist()
        except Exception:
            continue
        for real_value in real_values:
            if re.search(rf"\b{re.escape(str(real_value))}\b", working_text, re.IGNORECASE):
                filters.setdefault(col, {}).setdefault("values", [])
                if real_value not in filters[col]["values"]:
                    filters[col]["values"].append(real_value)

    # 3b. Implicit numeric equality — no "and"/"," connector at all (e.g.
    # "bmw car models in year 2023"). Without a connector, step 2 above
    # only ever finds ONE equality condition (it stops at the first match)
    # and silently drops the rest, which then fails the "2+ conditions"
    # gate below and falls through to the single-value lookup — which
    # picks one arbitrary matching row and reports it as "the" answer,
    # ignoring the other condition entirely.
    if len(clauses) == 1:
        scan_text = clauses[0]

        for value in extract_candidate_values(scan_text):
            if value in used_candidate_values:
                continue
            for col in numeric_cols:
                if col in filters:
                    continue
                try:
                    matches = find_matching_rows(df, col, value)
                except ValueError:
                    continue
                if not matches.empty:
                    actual_value = matches[col].iloc[0]
                    filters.setdefault(col, {}).setdefault("values", [])
                    if actual_value not in filters[col]["values"]:
                        filters[col]["values"].append(actual_value)
                    break

    if not filters:
        return None

    # Only worth using this path if there's a genuine range/threshold, or
    # more than one condition — a single plain equality is better handled
    # by the precise single-value lookup instead.
    has_range_or_threshold = any("min" in spec or "max" in spec for spec in filters.values())
    if not has_range_or_threshold and len(filters) < 2:
        return None

    return {"filters": filters}

# ---------------------------------------------------------------------------
# Distinct-values queries
#
# Everything above assumes the user wants a SPECIFIC ROW ("what is X where
# Y = Z", "cars between A and B"). But a question like "total 6 car brand
# names" or "list all fuel types" isn't asking for a row at all — it's
# asking for the SET OF DISTINCT VALUES a column takes. Previously there
# was no concept of this, so a number mentioned in passing (the "6" above,
# the user's own guess at the count) fell through to the single-value
# lookup path and got matched against an unrelated column (e.g. Car_ID),
# producing a confidently wrong specific answer instead of the actual list.
# ---------------------------------------------------------------------------

_DISTINCT_STRONG_TRIGGER_PATTERN = re.compile(
    r"\b(distinct|unique|different|list)\b", re.IGNORECASE
)
# "how many" / "total" are ambiguous — they just as often mean "sum this
# numeric column" (an aggregation this app doesn't support) as "list the
# distinct values" ("total 6 car brand names"). Only trust them for
# categorical columns, where "sum" doesn't make sense and "list the values"
# is the only sane reading — see the numeric-column check below.
_DISTINCT_WEAK_TRIGGER_PATTERN = re.compile(r"\b(how many|total)\b", re.IGNORECASE)


def detect_distinct_values_query(question: str, df: pd.DataFrame) -> dict | None:
    """
    Detects "what are the distinct/unique values in this column" questions
    and returns the real set of values found in the data. Deliberately
    narrow: bails out (returns None) if the question looks like an
    id-based row lookup ("id" mentioned), a range/threshold query (those
    are handled by detect_lookup_rule_based / detect_filter_query instead),
    or an aggregate-sounding request on a numeric column ("total horsepower
    of all BMW cars") that this app has no way to actually compute — so it
    never steals those cases or fabricates a meaningless answer.
    """
    strong_trigger = bool(_DISTINCT_STRONG_TRIGGER_PATTERN.search(question))
    weak_trigger = bool(_DISTINCT_WEAK_TRIGGER_PATTERN.search(question))
    if not strong_trigger and not weak_trigger:
        return None

    q_lower = question.lower()
    if re.search(r"\bid\b", q_lower):
        return None
    if _RANGE_PATTERN.search(question) or _MIN_PATTERN.search(question) or _MAX_PATTERN.search(question):
        return None

    target_col = guess_target_column(question, list(df.columns))
    if not target_col:
        return None
    if not strong_trigger and pd.api.types.is_numeric_dtype(df[target_col]):
        return None

    # If some OTHER column's real value is also mentioned ("Honda" in
    # "list all model_year of Honda"), this isn't a request for the WHOLE
    # column's distinct values — it's scoped to that condition ("what
    # years does Honda have"), which this detector has no way to express
    # (it only ever returns the unfiltered set). Silently ignoring the
    # condition and returning every Model_Year in the whole dataset would
    # be misleading, so defer instead — that lets it fall through to a
    # real SQL query, which CAN express "DISTINCT x WHERE y = z".
    for col in df.columns:
        if col == target_col or pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            real_values = df[col].dropna().unique().tolist()
        except Exception:
            continue
        for real_value in real_values:
            if re.search(rf"\b{re.escape(str(real_value))}\b", question, re.IGNORECASE):
                return None

    values = sorted(df[target_col].dropna().unique().tolist(), key=str)
    return {"intent": "distinct", "column": target_col, "values": values, "count": len(values)}


def resolve_structured_query(question: str, df: pd.DataFrame) -> dict | None:
    """
    Single entry point for the two single-dataframe detectors above,
    enforcing the priority they MUST be tried in: detect_filter_query
    first, detect_distinct_values_query only if that finds nothing.

    Why this order is load-bearing, not arbitrary: detect_filter_query
    only ever returns non-None when it has matched a condition against
    REAL data, so it can't misfire. detect_distinct_values_query fires on
    broad trigger words like "list" or "total" alone — so "list bmw and
    tesla with 4 doors" would otherwise get hijacked into "show me every
    distinct Doors value" (because "doors" is a real column name and
    "list" is a strong trigger) before the filter detector — which
    correctly resolves Brand IN (BMW, Tesla) AND Doors = 4 — ever got a
    chance to run. That's exactly the bug this function exists to make
    structurally impossible to reintroduce: this ordering lives here,
    covered by tests, instead of being re-derived (and easy to get
    backwards) at each call site.

    Returns {"kind": "filter", "filters": {...}} or
    {"kind": "distinct", "column": ..., "values": ..., "count": ...},
    or None if neither detector resolves the question.
    """
    filter_query = detect_filter_query(question, df)
    if filter_query:
        return {"kind": "filter", **filter_query}

    distinct_query = detect_distinct_values_query(question, df)
    if distinct_query:
        return {"kind": "distinct", **distinct_query}

    return None