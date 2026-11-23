"""Text normalisation and tokenisation (owned by Hassan Ul Haq).

We deliberately keep NLP in plain Python + regex so the pipeline is easy to
explain in a viva and has no heavy NLP dependency. The same functions are
plugged into scikit-learn's TfidfVectorizer, so training and prediction always
use identical preprocessing.
"""

from __future__ import annotations

import re
from typing import Final

# Small, hand-picked stopword list. We intentionally do NOT remove words such
# as "not", "no", "my", "where" or "when": in support tickets they carry
# meaning ("not received", "where is my order").
STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for",
        "with", "about", "to", "from", "in", "on", "is", "are", "was", "were",
        "be", "been", "being", "am", "do", "does", "did", "i", "me", "you",
        "your", "we", "us", "it", "its", "this", "that", "these", "those",
        "there", "here", "so", "too", "very", "just", "please", "pls", "plz",
        "hi", "hello", "hey", "thanks", "thank", "would", "could", "can",
        "will", "shall", "should", "im", "ive", "id", "want", "need", "like",
        "some", "any", "assistance", "help", "know", "tell", "let", "see",
    }
)

_URL_RE: Final[re.Pattern[str]] = re.compile(r"(https?://\S+|www\.\S+)")
_EMAIL_RE: Final[re.Pattern[str]] = re.compile(r"\S+@\S+\.\S+")
# Bitext uses {{Order Number}}-style placeholders; they are noise for routing.
_PLACEHOLDER_RE: Final[re.Pattern[str]] = re.compile(r"\{\{.*?\}\}")
_NON_ALPHA_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z\s]")
_SPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")


def normalize_text(text: object) -> str:
    """Lower-case a message and strip URLs, e-mails, placeholders, digits and punctuation.

    Args:
        text: Raw message. Non-strings (e.g. NaN from pandas) become "".

    Returns:
        A clean, single-spaced lower-case string (may be empty).
    """
    if not isinstance(text, str):
        return ""
    cleaned = text.lower().strip()
    cleaned = _URL_RE.sub(" ", cleaned)
    cleaned = _EMAIL_RE.sub(" ", cleaned)
    cleaned = _PLACEHOLDER_RE.sub(" ", cleaned)
    # Apostrophes are removed (not spaced) so "don't" -> "dont" stays one token.
    cleaned = cleaned.replace("'", "").replace("’", "")
    cleaned = _NON_ALPHA_RE.sub(" ", cleaned)
    return _SPACE_RE.sub(" ", cleaned).strip()


def tokenize(text: str) -> list[str]:
    """Split normalised text into tokens, dropping stopwords and 1-letter tokens.

    Args:
        text: Text already passed through :func:`normalize_text`.

    Returns:
        List of meaningful tokens.
    """
    return [tok for tok in text.split() if len(tok) > 1 and tok not in STOPWORDS]


def preprocess(text: object) -> list[str]:
    """Convenience wrapper: normalise then tokenise."""
    return tokenize(normalize_text(text))


def word_count(text: object) -> int:
    """Number of whitespace-separated words in the *normalised* text."""
    norm = normalize_text(text)
    return len(norm.split()) if norm else 0
