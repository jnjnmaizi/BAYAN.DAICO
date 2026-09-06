"""Versioned, deliberately conservative bilingual preprocessing for Bayan."""

from __future__ import annotations

import re
import unicodedata


PREPROC_VERSION = "1.2.0"

_TATWEEL = "\u0640"
_REPEATED_CHARACTERS = re.compile(r"(.)\1{2,}", flags=re.DOTALL)
_WHITESPACE = re.compile(r"\s+")

# A Saudi mobile number is 05xxxxxxxx locally, or +966/966/00966 followed by
# 5xxxxxxxx.  Separators are allowed so the same privacy rule applies to a
# number a citizen has formatted for readability.
_PHONE = re.compile(
    r"(?<![0-9])(?:"
    r"(?:\+966|00966|966)[\s.-]*5(?:[\s.-]*[0-9]){8}"
    r"|0?5(?:[\s.-]*[0-9]){8}"
    r")(?![0-9])"
)
_NATIONAL_ID = re.compile(r"(?<![0-9])[12][0-9]{9}(?![0-9])")


def _require_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Bayan preprocessing expects a string")
    return text


def normalize(text: str) -> str:
    """Return deterministic normalisation while retaining task-bearing text.

    We intentionally do not strip emoji, Arabic diacritics, case, or HTML-like
    tokens here: they can carry sentiment, meaning, or provenance.  The
    contract only folds Unicode representation, elongation noise, and spacing.
    """
    text = unicodedata.normalize("NFC", _require_text(text))
    text = text.replace(_TATWEEL, "")
    text = _REPEATED_CHARACTERS.sub(lambda match: match.group(1) * 2, text)
    return _WHITESPACE.sub(" ", text).strip()


def mask_pii(text: str) -> str:
    """Mask supported Saudi mobile numbers and national-ID-shaped values."""
    text = _require_text(text)
    # Phone first: an international phone can contain a 10-digit substring that
    # otherwise resembles an ID once country code formatting is removed.
    text = _PHONE.sub("<PHONE>", text)
    return _NATIONAL_ID.sub("<NATIONAL_ID>", text)


def preprocess(text: str) -> str:
    """Apply privacy masking before the shared normalisation contract."""
    return normalize(mask_pii(text))
