"""Audit four tokenizer candidates on Bayan Arabic and English feedback."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

CANDIDATES = {
    "bert-base-multilingual-cased": "mBERT",
    "xlm-roberta-base": "XLM-R",
    "CAMeL-Lab/bert-base-arabic-camelbert-mix": "CAMeLBERT",
    "distilbert-base-uncased": "DistilBERT",
}

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/bayan_feedback.csv"
REVISIONS = {
    "bert-base-multilingual-cased": "3f076fdb1ab68d5b2880cb87a0886f315b8146f8",
    "xlm-roberta-base": "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
    "CAMeL-Lab/bert-base-arabic-camelbert-mix": "9be352797bdf28a9ae21e2ae582aaaca7abdb22d",
    "distilbert-base-uncased": "12040accade4e8a0f71eabdb258fecc2e7e948be",
}


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


def tokenizer_statistics(tokenizer, texts: list[str]) -> dict:
    """Audit untruncated lengths; content pieces exclude model special tokens."""
    lengths = []
    unknown_pieces = 0
    total_pieces = 0
    words = sum(len(text.split()) for text in texts)
    for text in texts:
        content_ids = tokenizer(text, add_special_tokens=False, truncation=False).input_ids
        total_pieces += len(content_ids)
        if tokenizer.unk_token_id is not None:
            unknown_pieces += sum(piece == tokenizer.unk_token_id for piece in content_ids)
        lengths.append(len(tokenizer(text, add_special_tokens=True, truncation=False).input_ids))
    return {
        "n": len(texts), "fertility": total_pieces / words if words else 0.0,
        "p95_length": percentile_95(lengths),
        "unk_percent": 100 * unknown_pieces / total_pieces if total_pieces else 0.0,
        "lengths": lengths,
        "length_counts": dict(sorted(Counter(lengths).items())),
    }


def audit_tokenizer(tokenizer, texts: list[str]) -> tuple[float, float, float]:
    """Preserve the existing fertility/p95/UNK interface."""
    stats = tokenizer_statistics(tokenizer, texts)
    return stats["fertility"], stats["p95_length"], stats["unk_percent"]


def save_histograms(results: dict, output: Path):
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    maximum = max(max(stats[lang]["lengths"], default=0)
                  for stats in results.values() for lang in ["ar", "en"])
    bins = np.arange(-0.5, maximum + 1.5, 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for axis, (label, stats) in zip(axes.flat, results.items()):
        for lang, name, color in [("ar", "Arabic", "#1864ab"), ("en", "English", "#d65f00")]:
            values = stats[lang]["lengths"]
            if not values:
                continue
            axis.hist(values, bins=bins, weights=np.full(len(values), 100 / len(values)),
                      histtype="step", linewidth=1.8, color=color,
                      label=f"{name}: n={len(values):,}, p95={stats[lang]['p95_length']:.0f}")
            axis.axvline(stats[lang]["p95_length"], color=color, linestyle="--", alpha=0.65)
        axis.set_title(label)
        axis.set_xlabel("Sequence length (tokens, including special tokens)")
        axis.set_ylabel("Feedback rows within language (%)")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=9)
    fig.suptitle("Lab 1 — Token-length distributions by tokenizer and language", fontsize=15)
    fig.text(0.5, 0.02, "Same raw corpus; no truncation. One-token bins. Dashed lines mark p95. Each language sums to 100%.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/lab1")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    slices = load_language_slices()
    print(f"Corpus: {len(slices['ar'])} Arabic / {len(slices['en'])} English rows", flush=True)
    print("| Tokenizer | AR fertility | EN fertility | AR p95 len | EN p95 len | AR UNK rate |", flush=True)
    print("|---|---:|---:|---:|---:|---:|", flush=True)
    results = {}
    for checkpoint, label in CANDIDATES.items():
        options = {"revision": REVISIONS[checkpoint]}
        try:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, **options)
        except OSError:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, **options)
        ar = tokenizer_statistics(tokenizer, slices["ar"])
        en = tokenizer_statistics(tokenizer, slices["en"])
        results[label] = {"checkpoint": checkpoint, "revision": REVISIONS[checkpoint], "ar": ar, "en": en}
        print(f"| {label} | {ar['fertility']:.3f} | {en['fertility']:.3f} | "
              f"{ar['p95_length']:.1f} | {en['p95_length']:.1f} | {ar['unk_percent']:.2f}% |", flush=True)
    figure = args.output_dir / "token_length_histograms.png"
    save_histograms(results, figure)
    # Save exact bin counts instead of raw feedback or per-row text.
    for stats in results.values():
        for lang in ["ar", "en"]:
            del stats[lang]["lengths"]
    report = {
        "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "input_policy": "Unmodified raw text, matching the original Lab 1 audit; no truncation",
        "fertility_definition": "Content subword pieces / whitespace-delimited words; excludes special tokens",
        "histogram_definition": "One-token bins including special tokens; each language independently normalized to 100%",
        "tokenizers": results,
    }
    report_path = args.output_dir / "tokenizer_audit.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Saved chart: {figure}", flush=True)
    print(f"Saved numeric results: {report_path}", flush=True)


if __name__ == "__main__":
    main()
