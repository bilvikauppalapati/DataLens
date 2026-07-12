"""
chatbot.py — Phase 8

Uses a LOCAL LLM via Ollama instead of a paid API. No API key needed,
runs fully offline on your machine.

Requirements:
    1. Install Ollama: https://ollama.com
    2. Pull a model once:  ollama pull llama3.2:3b
    3. Make sure Ollama is running (it runs automatically in the background
       after install, serving at http://localhost:11434)

Design principle: NEVER send the full dataframe to the model. Send only:
    - schema (column names + dtypes)
    - profile_dataset() output from loaders.py (nulls, shape, memory)
    - summary_stats() output from stats.py
    - a small sample (e.g. df.head(5).to_dict())
    - transform_history from session_state (what preprocessing was applied)
This keeps prompts small and fast — important since local models are
slower than cloud APIs, especially on modest hardware like 8GB RAM.
"""
from __future__ import annotations

import json
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:3b"   # change this if you pull a different model

SYSTEM_PROMPT = (
    "You are a data analyst assistant embedded in a tool called DataLens. "
    "You will be given a dataset's schema, summary statistics, a small "
    "sample of rows, and a log of preprocessing steps already applied. "
    "Answer the user's question using ONLY this context. "
    "Be concise and specific. If the context doesn't contain the answer, "
    "say so instead of guessing."
)


def build_context(profile: dict, stats: dict | None, sample_rows: list[dict],
                   transform_history: list[dict]) -> str:
    """
    Packs everything the model needs into one compact text block.
    Keeping this small matters a lot for local models — smaller prompt =
    faster response on limited hardware.
    """
    parts = [
        "## Dataset profile",
        json.dumps(profile, indent=2, default=str),
    ]
    if stats:
        parts.append("## Summary statistics")
        parts.append(json.dumps(stats, indent=2, default=str))
    if sample_rows:
        parts.append("## Sample rows (first few)")
        parts.append(json.dumps(sample_rows, indent=2, default=str))
    if transform_history:
        parts.append("## Preprocessing already applied")
        parts.append(json.dumps(transform_history, indent=2, default=str))
    return "\n\n".join(parts)


def ask_chatbot(question: str, context: str, history: list[dict] | None = None) -> str:
    """
    Sends the question + context to the local Ollama server and returns
    the model's text reply.

    Raises:
        ConnectionError: if Ollama isn't running.
        RuntimeError: for any other API-level failure.
    """
    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nContext:\n{context}"}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "messages": messages, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(
            "Could not reach Ollama at http://localhost:11434. "
            "Is Ollama running? Try opening the Ollama app or running "
            "`ollama serve` in a terminal."
        ) from e
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e

    data = response.json()
    return data["message"]["content"]


def check_ollama_available() -> bool:
    """Quick health check — call this before showing the chat UI so we can
    show a friendly error instead of a stack trace if Ollama isn't running."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False
# ---------------------------------------------------------------------------
# Lookup intent detection
#
# The LLM's ONLY job here is to figure out WHAT the user is asking for.
# It never answers the lookup itself — that would risk it guessing a wrong
# value. Once we know the intent, core/lookup.py runs a real, exact pandas
# lookup and that real result is what gets shown/reported back.
# ---------------------------------------------------------------------------

LOOKUP_INTENT_SYSTEM_PROMPT = (
    "You classify a user's question about datasets into structured JSON. "
    "Respond with ONLY a JSON object, no other text, no markdown fences.\n\n"
    "If the question asks for a specific value looked up by matching a "
    "column to a value, in ONE dataset, respond with:\n"
    '{"intent": "lookup", "scope": "single", "dataset": "<dataset name>", '
    '"filter_column": "<column>", "filter_value": "<value>", "target_column": "<column>"}\n\n'
    "If the question asks for a value in ONE dataset based on matching a "
    "shared column's value in ANOTHER dataset, respond with:\n"
    '{"intent": "lookup", "scope": "cross", "dataset_a": "<dataset with the given value>", '
    '"join_column": "<shared column>", "filter_value": "<value>", '
    '"dataset_b": "<dataset to look the answer up in>", "target_column": "<column in dataset_b>"}\n\n'
    "If the question is NOT a specific value lookup (e.g. it's a general "
    "summary, statistics, or open-ended question), respond with:\n"
    '{"intent": "general"}\n\n'
    "Only use dataset and column names that are given to you below. Never "
    "invent a name that isn't listed."
)


def build_lookup_intent_prompt(question: str, dataset_columns: dict[str, list[str]]) -> str:
    """Builds the user-facing prompt text listing available datasets/columns,
    so the model only ever picks from real names instead of inventing one."""
    lines = ["Available datasets and their columns:"]
    for name, cols in dataset_columns.items():
        lines.append(f"- {name}: {', '.join(cols)}")
    lines.append("")
    lines.append(f"Question: {question}")
    return "\n".join(lines)


def parse_lookup_response(raw_text: str, dataset_columns: dict[str, list[str]]) -> dict | None:
    """
    Parses and validates the model's JSON response against the REAL
    datasets/columns we told it about. Returns None (meaning: fall back to
    general chat) if the JSON is malformed, incomplete, or references a
    dataset/column that doesn't actually exist — this stops a hallucinated
    column name from ever reaching the real lookup code.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(parsed, dict):
        return None

    intent = parsed.get("intent")
    if intent == "general":
        return {"intent": "general"}

    if intent != "lookup":
        return None

    scope = parsed.get("scope")

    if scope == "single":
        dataset = parsed.get("dataset")
        filter_column = parsed.get("filter_column")
        filter_value = parsed.get("filter_value")
        target_column = parsed.get("target_column")

        if not all([dataset, filter_column, filter_value is not None, target_column]):
            return None
        if dataset not in dataset_columns:
            return None
        cols = dataset_columns[dataset]
        if filter_column not in cols or target_column not in cols:
            return None

        return {
            "intent": "lookup", "scope": "single", "dataset": dataset,
            "filter_column": filter_column, "filter_value": filter_value,
            "target_column": target_column,
        }

    elif scope == "cross":
        dataset_a = parsed.get("dataset_a")
        dataset_b = parsed.get("dataset_b")
        join_column = parsed.get("join_column")
        filter_value = parsed.get("filter_value")
        target_column = parsed.get("target_column")

        if not all([dataset_a, dataset_b, join_column, filter_value is not None, target_column]):
            return None
        if dataset_a not in dataset_columns or dataset_b not in dataset_columns:
            return None
        if join_column not in dataset_columns[dataset_a] or join_column not in dataset_columns[dataset_b]:
            return None
        if target_column not in dataset_columns[dataset_b]:
            return None

        return {
            "intent": "lookup", "scope": "cross", "dataset_a": dataset_a, "dataset_b": dataset_b,
            "join_column": join_column, "filter_value": filter_value, "target_column": target_column,
        }

    return None


def detect_lookup_intent(question: str, dataset_columns: dict[str, list[str]]) -> dict | None:
    """
    Asks the local LLM to classify the question, then validates the
    response against the real dataset/column names. Returns a dict
    describing the lookup (see parse_lookup_response), {"intent": "general"}
    for non-lookup questions, or None if the model's response couldn't be
    understood/validated (caller should fall back to general chat in that case).
    """
    prompt = build_lookup_intent_prompt(question, dataset_columns)
    messages = [
        {"role": "system", "content": LOOKUP_INTENT_SYSTEM_PROMPT},
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

    raw_text = response.json().get("message", {}).get("content", "")
    return parse_lookup_response(raw_text, dataset_columns)