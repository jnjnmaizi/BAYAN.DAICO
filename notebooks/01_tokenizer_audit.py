"""Audit four tokenizer candidates on Bayan Arabic and English feedback."""
from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

CANDIDATES = {
    "bert-base-multilingual-cased": "mBERT",
    "xlm-roberta-base": "XLM-R",
    "CAMeL-Lab/bert-base-arabic-camelbert-mix": "CAMeLBERT",
    "distilbert-base-uncased": "DistilBERT",
}

DATA = Path("data/raw/bayan_feedback.csv")


def fertility(tokenizer, texts: Iterable[str]) -> float:
    """Return content subword pieces per whitespace-delimited word."""
    pieces = 0
    words = 0
    for text in texts:
        words += len(text.split())
        pieces += len(tokenizer(text, add_special_tokens=False).input_ids)
    return pieces / words if words else 0.0


def percentile_95(values: list[int]) -> float:
    """Compute the linear-interpolated 95th percentile without numpy."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def load_language_slices() -> dict[str, list[str]]:
    """Load the AR/EN corpus slices while tolerating the CSV UTF-8 BOM."""
    slices = {"ar": [], "en": []}
    with DATA.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            language = row["lang"].lower()
            if language in slices and row["text"].strip():
                slices[language].append(row["text"])
    return slices


def audit_tokenizer(tokenizer, texts: list[str]) -> tuple[float, float, float]:
    """Return fertility, p95 sequence length, and content UNK percentage."""
    lengths = []
    unknown_pieces = 0
    total_pieces = 0
    for text in texts:
        content_ids = tokenizer(text, add_special_tokens=False).input_ids
        total_pieces += len(content_ids)
        if tokenizer.unk_token_id is not None:
            unknown_pieces += sum(piece == tokenizer.unk_token_id for piece in content_ids)
        lengths.append(len(tokenizer(text, add_special_tokens=True).input_ids))
    unk_rate = 100 * unknown_pieces / total_pieces if total_pieces else 0.0
    return fertility(tokenizer, texts), percentile_95(lengths), unk_rate


def main():
    from transformers import AutoTokenizer

    slices = load_language_slices()
    print(f"Corpus: {len(slices['ar'])} Arabic / {len(slices['en'])} English rows")
    print("| Tokenizer | AR fertility | EN fertility | AR p95 len | EN p95 len | AR UNK rate |")
    print("|---|---:|---:|---:|---:|---:|")

    for checkpoint, label in CANDIDATES.items():
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        ar_fertility, ar_p95, ar_unk = audit_tokenizer(tokenizer, slices["ar"])
        en_fertility, en_p95, _ = audit_tokenizer(tokenizer, slices["en"])
        print(
            f"| {label} | {ar_fertility:.3f} | {en_fertility:.3f} | "
            f"{ar_p95:.1f} | {en_p95:.1f} | {ar_unk:.2f}% |"
        )


if __name__ == "__main__":
    main()
