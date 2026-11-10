"""
chatbot.py — Phase 8

Uses the Google Gemini API (cloud) for all LLM calls — no local server needed.

Requirements:
    1. Get an API key: https://aistudio.google.com/app/apikey
    2. Set it as GEMINI_API_KEY — either in your shell, in a `.env` file at
       the project root, or in `.streamlit/secrets.toml`.
    3. Optional: set GEMINI_MODEL to override the default model.

Design principle: NEVER send the full dataframe to the model. Send only:
    - schema (column names + dtypes)
    - profile_dataset() output from loaders.py (nulls, shape, memory)
    - summary_stats() output from stats.py
    - a small sample (e.g. df.head(5).to_dict())
    - transform_history from session_state (what preprocessing was applied)
This keeps prompts small, fast and cheap, and avoids uploading user data.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-3.5-flash"


def _load_dotenv() -> None:
    """Minimal .env loader (KEY=VALUE lines) so python-dotenv isn't required.
    Never overrides variables already set in the environment."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)


def get_api_key() -> str | None:
    """Looks up the Gemini key in the environment (incl. .env), then in
    Streamlit secrets if running inside Streamlit."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        return None


def check_llm_available() -> bool:
    """Quick check — call this before showing the chat UI so we can show a
    friendly error instead of a stack trace if no API key is configured."""
    return bool(get_api_key())


def call_llm(messages: list[dict], timeout: int = 60) -> str:
    """
    Sends chat-style messages ({"role": "system"|"user"|"assistant",
    "content": str}) to Gemini and returns the reply text.

    Raises:
        ConnectionError: if no API key is configured or Gemini can't be reached.
        RuntimeError: for any other API-level failure.
    """
    api_key = get_api_key()
    if not api_key:
        raise ConnectionError(
            "No Gemini API key found. Set GEMINI_API_KEY in your environment, "
            "a .env file, or .streamlit/secrets.toml."
        )

    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    contents = [
        {
            "role": "model" if m["role"] == "assistant" else "user",
            "parts": [{"text": m["content"]}],
        }
        for m in messages
        if m["role"] != "system"
    ]
    payload: dict = {"contents": contents}
    if system_parts:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}

    try:
        response = requests.post(
            f"{GEMINI_API_BASE}/{MODEL_NAME}:generateContent",
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError("Could not reach the Gemini API. Check your internet connection.") from e
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("error", {}).get("message", "")
        except Exception:
            pass
        raise RuntimeError(f"Gemini request failed: {e}. {detail}".strip()) from e
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Gemini request failed: {e}") from e

    data = response.json()
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError, TypeError):
        reason = data.get("promptFeedback", {}).get("blockReason")
        raise RuntimeError(
            f"Gemini returned no answer{f' (blocked: {reason})' if reason else ''}."
        )


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
    Keeping this small keeps responses fast and token costs low.
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
    Sends the question + context to Gemini and returns
    the model's text reply.

    Raises:
        ConnectionError: if no API key is set or Gemini is unreachable.
        RuntimeError: for any other API-level failure.
    """
    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nContext:\n{context}"}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    return call_llm(messages, timeout=120)


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
    Asks the LLM to classify the question, then validates the
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
        raw_text = call_llm(messages, timeout=60)
    except (ConnectionError, RuntimeError):
        return None

    return parse_lookup_response(raw_text, dataset_columns)