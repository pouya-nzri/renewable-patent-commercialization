"""
data.py
=======
Data loading, text cleaning, and structured-feature engineering.

Implements the paper's data preparation exactly:
  * six textual fields, cleaned (lowercase + regex + stopword removal)   [paper]
  * five structured count indicators                                      [paper]
  * NO target leakage: the dependent variable is never placed in the
    feature space, and structured standardization is deferred to the
    modeling stage so that scaler moments come from TRAIN folds only.

The cleaning function reproduces the paper's description
("lowercase + regex + stopwords").
"""

from __future__ import annotations
import json
import re
from typing import List, Tuple

import numpy as np
import pandas as pd

import config as C

try:
    import nltk
    from nltk.corpus import stopwords
    try:
        _STOP = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        _STOP = set(stopwords.words("english"))
except Exception:  # pragma: no cover - fallback if nltk unavailable
    _STOP = {
        "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with",
        "is", "are", "be", "by", "as", "at", "that", "this", "it", "from",
    }

_WORD_RE = re.compile(r"[\W_]+")
_DIGIT_RE = re.compile(r"\d+")


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/underscores/digits, drop stopwords. [paper]"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _WORD_RE.sub(" ", text)
    text = _DIGIT_RE.sub("", text)
    tokens = [w for w in text.split() if w not in _STOP]
    return " ".join(tokens)


def _as_text(value) -> str:
    """Join list-valued JSON fields into a single string before cleaning."""
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    if isinstance(value, str):
        return value
    return ""


def load_stratum(tau: int, path: str | None = None,
                 verify_n: bool = False) -> pd.DataFrame:
    """
    Load one stratum (D{tau}) from JSON into a DataFrame.

    The JSON is expected as {patent_id: {fields...}}, matching the user's
    existing ./data/{tau}input.json layout.
    """
    path = path or C.PATHS.data_file(tau)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for pid, details in data.items():
        d = dict(details)
        d["patent_id"] = pid
        rows.append(d)
    df = pd.DataFrame(rows)

    if verify_n and tau in C.EXPECTED_N:
        # A soft check only: warns, never mutates data.
        if len(df) != C.EXPECTED_N[tau]:
            print(f"[warn] D{tau}: loaded N={len(df)} but paper reports "
                  f"N={C.EXPECTED_N[tau]}. Proceeding with the real data.")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the five structured indicators and the six cleaned text fields.

    Raw source columns (from the patent JSON):
        inventors, original_assignee, abstract, country_status,
        claims, backward_references, backward_references_assignee
    """
    def _len(col):
        return df[col].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # ---- structured count indicators (five) --------------------------------
    df["num_inventors"] = _len("inventors")
    df["num_assignees"] = _len("original_assignee")
    df["num_countries"] = _len("country_status")
    df["num_claims"] = df["claims"].apply(
        lambda x: _count_claims(x))
    df["num_backward_refs"] = _len("backward_references")

    # ---- six cleaned text fields ------------------------------------------
    df["assignees_text"] = df["original_assignee"].apply(lambda x: clean_text(_as_text(x)))
    df["claims_text"] = df["claims"].apply(lambda x: clean_text(_as_text(x)))
    df["countries_text"] = df["country_status"].apply(lambda x: clean_text(_as_text(x)))
    df["backward_refs_text"] = df["backward_references"].apply(lambda x: clean_text(_as_text(x)))
    df["inventors_text"] = df["inventors"].apply(lambda x: clean_text(_as_text(x)))
    df["abstract_text"] = df["abstract"].apply(lambda x: clean_text(_as_text(x)))

    # Raw assignee string retained for cold-start grouping and memorization
    # baselines (NOT used as a model feature).
    df["assignee_key"] = df["original_assignee"].apply(_primary_assignee)

    return df


def _count_claims(claims_value) -> int:
    """
    Number of claims. Claims often arrive as a single long string with
    enumerated items ("1. ... 2. ..."); fall back to list length.
    """
    if isinstance(claims_value, list):
        if len(claims_value) == 1 and isinstance(claims_value[0], str):
            return _count_enumerated(claims_value[0])
        return len(claims_value)
    if isinstance(claims_value, str):
        return _count_enumerated(claims_value)
    return 0


def _count_enumerated(text: str) -> int:
    # Count "N." claim markers; at least 1 if any text present.
    markers = re.findall(r"(?:^|\s)(\d{1,3})\.\s", text)
    return max(len(markers), 1 if text.strip() else 0)


def _primary_assignee(value) -> str:
    if isinstance(value, list) and value:
        return str(value[0]).strip().lower()
    if isinstance(value, str):
        return value.strip().lower()
    return "__unknown__"


def get_target(df: pd.DataFrame) -> np.ndarray:
    """Return the continuous target, guaranteeing it never enters features."""
    if C.TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{C.TARGET_COLUMN}' missing from data.")
    return df[C.TARGET_COLUMN].astype(float).values


def structured_matrix(df: pd.DataFrame) -> np.ndarray:
    """Raw (unscaled) structured indicators; scaling happens per-fold."""
    return df[C.STRUCTURED_FIELDS].astype(float).values
