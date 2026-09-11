"""Run an independent TF-IDF topic baseline on ArBNTopic.

This is a separate 14-class Arabic topic benchmark. It does not reuse Bayan
labels and never overwrites the official topic artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "secondary" / "arbn_topic_eval.json"


def main() -> None:
    try:
        ds = load_dataset("U4RASD/ArBNTopic")
    except Exception as exc:
        report = {
            "dataset": "U4RASD/ArBNTopic",
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

    train_text = ds["train"]["text"]
    train_y = ds["train"]["en_topic"]
    test_text = ds["test"]["text"]
    test_y = ds["test"]["en_topic"]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    x_train = vectorizer.fit_transform(train_text)
    x_test = vectorizer.transform(test_text)
    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    model.fit(x_train, train_y)
    pred = model.predict(x_test)
    report = {
        "dataset": "U4RASD/ArBNTopic",
        "download_status": "loaded",
        "train_rows": len(train_text),
        "test_rows": len(test_text),
        "label_count": len(set(train_y)),
        "accuracy": float(accuracy_score(test_y, pred)),
        "macro_f1": float(f1_score(test_y, pred, average="macro", zero_division=0)),
        "official_bayan_metrics_modified": False,
        "interpretation": "Independent external benchmark only; this score is not a replacement for Bayan Lab 3's eight-label frozen-test result.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
