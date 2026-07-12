"""
loaders.py — Phase 1
Responsible for reading uploaded files into pandas DataFrames and
producing a lightweight metadata summary for the preview screen.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


SUPPORTED_EXTENSIONS = {"csv", "xlsx", "xls", "json", "parquet"}


@dataclass
class LoadResult:
    """Wraps a loaded dataframe with metadata + any warnings raised while loading."""
    dataframe: pd.DataFrame
    filename: str
    warnings: list[str] = field(default_factory=list)


def get_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def load_dataset(file_bytes: bytes, filename: str) -> LoadResult:
    """
    Main entrypoint for Phase 1. Takes raw bytes + filename (as given by
    st.file_uploader) and returns a LoadResult.

    Raises:
        ValueError: if the extension isn't supported or the file can't be parsed.
    """
    ext = get_extension(filename)
    warnings: list[str] = []

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '.{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    buffer = io.BytesIO(file_bytes)

    if ext == "csv":
        df, warnings = _load_csv(buffer)
    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(buffer)
    elif ext == "json":
        df = _load_json(buffer)
    elif ext == "parquet":
        df = pd.read_parquet(buffer)
    else:  # pragma: no cover - guarded above
        raise ValueError(f"Unhandled extension: {ext}")

    return LoadResult(dataframe=df, filename=filename, warnings=warnings)


def _load_csv(buffer: io.BytesIO) -> tuple[pd.DataFrame, list[str]]:
    """Try a couple of encodings/delimiters before giving up. Returns (df, warnings)."""
    warnings: list[str] = []
    try:
        df = pd.read_csv(buffer)
        return df, warnings
    except UnicodeDecodeError:
        warnings.append("UTF-8 decoding failed; retried with latin-1 encoding.")
        buffer.seek(0)
        df = pd.read_csv(buffer, encoding="latin-1")
        return df, warnings
    except pd.errors.ParserError:
        warnings.append("Default comma delimiter failed; retried with delimiter auto-detection.")
        buffer.seek(0)
        df = pd.read_csv(buffer, sep=None, engine="python")
        return df, warnings


def _load_json(buffer: io.BytesIO) -> pd.DataFrame:
    raw = json.load(buffer)
    # Handle both {"records": [...]} style and plain list-of-dicts style JSON.
    if isinstance(raw, dict) and len(raw) == 1:
        only_value = next(iter(raw.values()))
        if isinstance(only_value, list):
            raw = only_value
    return pd.json_normalize(raw)


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """
    Produces the metadata shown on the Preview screen:
    shape, dtypes, null counts, memory usage.
    """
    return {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": df.isnull().sum().to_dict(),
        "null_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3),
        "duplicate_rows": int(df.duplicated().sum()),
    }
