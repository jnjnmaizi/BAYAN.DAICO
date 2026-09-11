"""Audit external datasets for the secondary evaluation track.

This script never edits the official Bayan data or evaluation artifacts. It
records dataset metadata and small streaming samples only.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "secondary" / "dataset_audit.json"


def split_sizes(dataset) -> dict[str, int | None]:
    # DatasetDict exposes split datasets directly; `.info` belongs to a
    # DatasetBuilder and is not available on the loaded mapping.
    return {name: len(split) for name, split in dataset.items()}


def main() -> None:
    result: dict[str, object] = {
        "created": str(date.today()),
        "purpose": "Secondary external-data evaluation; official Bayan metrics are unchanged",
        "official_data_modified": False,
        "datasets": {},
    }

    result["datasets"]["ArBNTopic"] = {
        "source": "https://huggingface.co/datasets/U4RASD/ArBNTopic",
        "license": "See the dataset card before redistribution",
        "task": "Arabic topic classification",
        "published_size": 19784,
        "published_label_count": 14,
        "role": "Independent topic coverage check for Lab 3; not mapped to Bayan's eight labels",
    }
    try:
        topic = load_dataset("U4RASD/ArBNTopic")
        topic_labels = Counter(topic["train"]["en_topic"])
        result["datasets"]["ArBNTopic"].update({
            "download_status": "loaded",
            "splits": split_sizes(topic),
            "train_label_count": len(topic_labels),
            "train_label_support": dict(sorted(topic_labels.items())),
        })
    except Exception as exc:
        result["datasets"]["ArBNTopic"].update({
            "download_status": "not_loaded_in_current_environment",
            "download_error": f"{type(exc).__name__}: {exc}",
        })

    result["datasets"]["Alyah"] = {
        "source": "https://huggingface.co/datasets/tiiuae/alyah-emirati-benchmark",
        "license": "See the dataset card before redistribution",
        "task": "Emirati dialect multiple-choice evaluation",
        "published_test_size": 1173,
        "role": "Gulf/Emirati robustness evaluation for Lab 4; not substituted for LOCATION recall",
    }
    try:
        emirati = load_dataset("tiiuae/alyah-emirati-benchmark")
        categories = Counter(emirati["test"]["category"])
        result["datasets"]["Alyah"].update({
            "download_status": "loaded",
            "splits": split_sizes(emirati),
            "category_count": len(categories),
            "category_support": dict(sorted(categories.items())),
        })
    except Exception as exc:
        result["datasets"]["Alyah"].update({
            "download_status": "not_loaded_in_current_environment",
            "download_error": f"{type(exc).__name__}: {exc}",
        })

    ner_rows = 0
    ner_loc = 0
    ner_label_counts: Counter[str] = Counter()
    ner_error = None
    try:
        ner_stream = load_dataset("iahlt/arabic_ner_mafat", split="train", streaming=True)
        for row in ner_stream.take(2000):
            ner_rows += 1
            for span in row.get("spans", []) or []:
                label = str(span.get("label", ""))
                ner_label_counts[label] += 1
                if label in {"LOC", "GPE", "LOCATION"}:
                    ner_loc += 1
    except Exception as exc:
        ner_error = f"{type(exc).__name__}: {exc}"
    result["datasets"]["IAHLT Arabic NER"] = {
        "source": "https://huggingface.co/datasets/iahlt/arabic_ner_mafat",
        "license": "See the dataset card before redistribution",
        "task": "Arabic span and token NER",
        "published_rows": 40000,
        "download_status": "loaded" if ner_error is None else "not_loaded_in_current_environment",
        "stream_sample_rows": ner_rows,
        "stream_sample_location_like_spans": ner_loc,
        "stream_sample_span_labels": dict(sorted(ner_label_counts.items())),
        "role": "Independent Arabic NER/location robustness check for Lab 4",
    }
    if ner_error:
        result["datasets"]["IAHLT Arabic NER"]["download_error"] = ner_error

    rag_rows = 0
    rag_dialects: Counter[str] = Counter()
    rag_categories: Counter[str] = Counter()
    rag_passages: set[str] = set()
    rag_error = None
    try:
        rag_stream = load_dataset("HeshamHaroon/ArabicRAGB", split="train", streaming=True)
        for row in rag_stream.take(2000):
            rag_rows += 1
            rag_dialects[str(row.get("query_dialect", "unknown"))] += 1
            rag_categories[str(row.get("source_category", "unknown"))] += 1
            rag_passages.add(str(row.get("passage_id", "")))
    except Exception as exc:
        rag_error = f"{type(exc).__name__}: {exc}"
    result["datasets"]["ArabicRAGB"] = {
        "source": "https://huggingface.co/datasets/HeshamHaroon/ArabicRAGB",
        "license": "CC BY-SA 4.0 according to the dataset card",
        "task": "Arabic passage-grounded retrieval",
        "published_rows": 13163,
        "download_status": "loaded" if rag_error is None else "not_loaded_in_current_environment",
        "stream_sample_rows": rag_rows,
        "stream_sample_unique_passages": len(rag_passages),
        "stream_sample_query_dialects": dict(sorted(rag_dialects.items())),
        "stream_sample_source_categories": dict(sorted(rag_categories.items())),
        "role": "Independent retrieval benchmark with explicit query-positive passage pairs for Lab 5",
    }
    if rag_error:
        result["datasets"]["ArabicRAGB"]["download_error"] = rag_error

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
