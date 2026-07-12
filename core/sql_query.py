"""
sql_query.py — Phase 8 extension
Text-to-SQL: ask the local LLM to translate a natural-language question
into a real SQL SELECT statement, then run that statement against the
ACTUAL data (an in-memory SQLite database built fresh from the dataframe)
and return the real result — no guessing, no 5-row sample.

Why this exists: core.nlq's rule-based detectors and core.chatbot's
detect_lookup_intent (a fixed "one column equals one value" JSON schema)
can only ever express what someone already anticipated and coded for —
every new phrasing (a range, two OR'd values, an aggregate like SUM/COUNT,
a GROUP BY) needed its own hand-written detector. SQL is expressive enough
to state the question directly, so instead of extending the JSON schema
forever, ask the model for SQL and actually run it.

Safety: only a single read-only SELECT (or WITH ... SELECT) statement is
ever executed, against a throwaway in-memory SQLite database that's
populated from the dataframe and discarded when the connection closes —
never the user's real file, never a persistent database, and no
INSERT/UPDATE/DELETE/DDL is allowed regardless of what the model outputs.
"""
from __future__ import annotations

import os
import re
import sqlite3

import pandas as pd
import requests

from core.chatbot import OLLAMA_URL, MODEL_NAME


SQL_SYSTEM_PROMPT = (
    "You are a SQL generator for a data analysis tool. You will be given "
    "one or more SQLite table names, each with its column names and "
    "types, and a user's question in plain English. Respond with ONLY a "
    "single valid SQLite SELECT statement (or a WITH ... SELECT) that "
    "answers the question — no explanation, no markdown fences, no other "
    "text before or after it. Only reference the exact table and column "
    "names given to you; never invent one, and never add a file "
    "extension to a table name. If the question needs data from more "
    "than one table, JOIN them on a shared column. Never write INSERT/"
    "UPDATE/DELETE/DROP/ALTER/ATTACH/PRAGMA — read-only SELECT "
    "statements only, and only ONE statement (no semicolon-separated "
    "chains). If the question genuinely can't be answered with a single "
    "SELECT over these tables, respond with exactly: NONE"
)

_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|"
    r"vacuum|replace|grant|reindex)\b",
    re.IGNORECASE,
)


def build_table_aliases(datasets: dict[str, pd.DataFrame]) -> dict[str, str]:
    """
    Maps each real dataset name to a SQL-safe bare identifier (letters,
    digits, underscores only, extension stripped) to use as the actual
    SQLite table name. A real filename like "cars.csv" contains a dot,
    which SQLite's parser treats as a schema.table separator unless the
    whole identifier is quoted — and small local models are unreliable
    about remembering to quote it (or keep the extension at all), so
    tables are aliased to names that never need quoting, sidestepping the
    whole class of "no such table" failures instead of trying to police
    the model's quoting after the fact.
    """
    aliases: dict[str, str] = {}
    used: set[str] = set()
    for name in datasets:
        base = re.sub(r"[^0-9a-zA-Z_]", "_", os.path.splitext(name)[0]) or "t"
        if base[0].isdigit():
            base = f"t_{base}"
        candidate = base
        i = 1
        while candidate in used:
            i += 1
            candidate = f"{base}_{i}"
        used.add(candidate)
        aliases[name] = candidate
    return aliases


def build_sql_prompt(
    question: str, datasets: dict[str, pd.DataFrame], aliases: dict[str, str] | None = None
) -> str:
    """Builds the user-facing prompt text: every table's (aliased) name
    and real column names/dtypes, and the question — so the model only
    ever references real identifiers instead of inventing one."""
    aliases = aliases or build_table_aliases(datasets)
    lines = []
    for name, df in datasets.items():
        lines.append(f"Table {aliases[name]} (from dataset '{name}'):")
        for col, dtype in df.dtypes.items():
            lines.append(f"  - {col} ({dtype})")
    lines.append("")
    lines.append(f"Question: {question}")
    return "\n".join(lines)


def is_safe_select(sql: str) -> bool:
    """
    True only for a single, read-only SELECT/WITH statement: rejects
    multiple semicolon-separated statements (a classic SQL-injection
    vector), any DDL/DML keyword, and anything that doesn't start with
    SELECT or WITH.
    """
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        return False
    if ";" in stripped:
        return False
    if _FORBIDDEN_KEYWORDS.search(stripped):
        return False
    return bool(re.match(r"^(select|with)\b", stripped, re.IGNORECASE))


def clean_sql_response(raw_text: str) -> str:
    """Strips markdown fences the model sometimes wraps the query in."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("sql"):
            text = text[3:]
        text = text.strip()
    return _normalize_sql_quoting(text.strip().rstrip(";").strip())


def _normalize_sql_quoting(sql: str) -> str:
    """
    Small local models occasionally mix SQL quoting dialects and wrap an
    already double-quoted identifier in backticks too — e.g.
    `` `"cars.csv"` `` — which SQLite parses as a table literally named
    `"cars.csv"` (quote characters included), not the real table `cars.csv`,
    so the query fails with "no such table" even though it was otherwise
    correct. Strips that redundant outer layer either direction.
    """
    sql = re.sub(r'`("[^"]*")`', r"\1", sql)
    sql = re.sub(r'"(`[^`]*`)"', r"\1", sql)
    return sql


def generate_sql(
    question: str, datasets: dict[str, pd.DataFrame], aliases: dict[str, str] | None = None
) -> str | None:
    """
    Asks the local LLM to translate the question into SQL over one or more
    tables (referred to by their SQL-safe aliases — see
    build_table_aliases). Returns the raw SQL string, or None if the
    model declined ("NONE"), errored, or produced something that fails
    the is_safe_select check — callers should fall back to the general
    chat path in that case.
    """
    aliases = aliases or build_table_aliases(datasets)
    prompt = build_sql_prompt(question, datasets, aliases)
    messages = [
        {"role": "system", "content": SQL_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "messages": messages, "stream": False},
            timeout=60,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException:
        return None

    raw = response.json().get("message", {}).get("content", "")
    sql = clean_sql_response(raw)

    if not sql or sql.upper() == "NONE":
        return None
    if not is_safe_select(sql):
        return None
    return sql


def run_sql(sql: str, table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Single-table convenience wrapper around run_sql_multi."""
    return run_sql_multi(sql, {table_name: df})


def run_sql_multi(sql: str, datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Executes `sql` against a throwaway in-memory SQLite database loaded
    with every dataframe in `datasets`, each under its own table name
    (letting the query JOIN across them). The connection — and the copy
    of the data inside it — is discarded when this function returns.

    Raises ValueError if `sql` fails the safety check or errors during
    execution (bad column/table name, syntax error, etc.) so the caller
    can fall back gracefully instead of crashing.
    """
    sql = _normalize_sql_quoting(sql)
    if not is_safe_select(sql):
        raise ValueError("Only a single read-only SELECT statement is allowed")

    conn = sqlite3.connect(":memory:")
    try:
        for table_name, df in datasets.items():
            df.to_sql(table_name, conn, index=False)
        try:
            return pd.read_sql_query(sql, conn)
        except (sqlite3.Error, pd.errors.DatabaseError) as e:
            raise ValueError(f"SQL execution failed: {e}") from e
    finally:
        conn.close()


def answer_with_sql(question: str, datasets: dict[str, pd.DataFrame]) -> tuple[str, pd.DataFrame] | None:
    """
    Full pipeline: generate SQL for `question` over every table in
    `datasets` (using SQL-safe aliases so the model never has to quote a
    dotted filename), run it for real, and return
    (sql_used, result_dataframe). Returns None if generation declined/
    failed, or if the generated query didn't actually execute — the
    caller should fall back to the general chat path.
    """
    aliases = build_table_aliases(datasets)
    sql = generate_sql(question, datasets, aliases=aliases)
    if sql is None:
        return None
    aliased_datasets = {alias: datasets[name] for name, alias in aliases.items()}
    try:
        result = run_sql_multi(sql, aliased_datasets)
    except ValueError:
        return None
    return sql, result
