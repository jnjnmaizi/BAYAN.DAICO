"""Evaluate a small, reproducible TF-IDF baseline on ArabicRAGB.

The benchmark is independent of the synthetic Bayan case IDs. Each sampled
record supplies one positive passage_id, so recall@10 and MRR@10 are defined
without borrowing the official labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "secondary" / "arabicragb_retrieval.json"


def main() -> None:
    try:
        rows = list(load_dataset("HeshamHaroon/ArabicRAGB", split="train", streaming=True).take(500))
    except Exception as exc:
        report = {
            "dataset": "HeshamHaroon/ArabicRAGB",
            "download_status": "not_loaded_in_current_environment",
            "download_error": f"{type(exc).__name__}: {exc}",
            "official_bayan_metrics_modified": False,
            "next_step": "Run this script in an environment with Hugging Face network access.",
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"Saved: {OUT}")
        return
    passages: dict[str, str] = {}
    for row in rows:
        passages.setdefault(str(row["passage_id"]), str(row["passage_text"]))
    passage_ids = list(passages)
    texts = [passages[pid] for pid in passage_ids]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    hits = []
    for row in rows:
        query = str(row["query"])
        positive = str(row["passage_id"])
        scores = (matrix @ vectorizer.transform([query]).T).toarray().ravel()
        order = np.argsort(-scores, kind="stable")[:10]
        ranked = [passage_ids[int(i)] for i in order]
        if positive in ranked:
            rank = ranked.index(positive) + 1
            hits.append((1, 1.0 / rank))
        else:
            hits.append((0, 0.0))
    report = {
        "dataset": "HeshamHaroon/ArabicRAGB",
        "split": "train",
        "sample_rows": len(rows),
        "unique_passages": len(passages),
        "metric_definition": "one supplied positive passage per query; char TF-IDF top-10",
        "recall_at_10": float(np.mean([x[0] for x in hits])) if hits else 0.0,
        "mrr_at_10": float(np.mean([x[1] for x in hits])) if hits else 0.0,
        "official_bayan_metrics_modified": False,
        "limitations": "Small streaming sample and a lexical baseline; this is an independent external benchmark, not a replacement for the official Bayan exact-ID score.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
