"""Language-neutral spaCy sentence segmentation used by the Bayan pipeline."""

from __future__ import annotations

import re

from .core import preprocess


_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s*$")
_ABBREVIATION_END = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e|a\.m|p\.m)\.$",
    flags=re.IGNORECASE,
)


def build_pipeline():
    """Build a lightweight, model-free pipeline for Arabic and English text."""
    import spacy

    nlp = spacy.blank("xx")
    nlp.add_pipe("sentencizer", config={"punct_chars": [".", "!", "?", "؟"]})
    return nlp


def split_sentences(raw: str, nlp) -> list[str]:
    """Preprocess *raw* and return its non-empty sentence strings.

    The sentencizer has high recall for terminal punctuation.  We then repair
    two predictable false breaks in feedback: numbered-list markers and common
    English abbreviations (including ``p.m.``).
    """
    if nlp is None:
        raise ValueError("Pass the spaCy pipeline returned by build_pipeline()")
    doc = nlp(preprocess(raw))
    candidates = [sentence.text.strip() for sentence in doc.sents if sentence.text.strip()]
    sentences: list[str] = []
    for candidate in candidates:
        if sentences and (_LIST_MARKER.fullmatch(sentences[-1]) or _ABBREVIATION_END.search(sentences[-1])):
            sentences[-1] = f"{sentences[-1]} {candidate}"
        else:
            sentences.append(candidate)
    return sentences
