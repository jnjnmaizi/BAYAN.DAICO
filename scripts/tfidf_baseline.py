"""Train the fixed TF-IDF baseline; frozen test is an explicit, one-shot step."""
import argparse
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from bayan.models.data import TOPIC_DATA, build_topic_dataset
from bayan.models.training import EVIDENCE, ROOT, sha256, split_evidence, topic_metrics, write_json
from bayan.preprocessing.core import PREPROC_VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/tfidf_baseline")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    test_path = args.output_dir / "frozen_test.json"
    if test_path.exists():
        raise SystemExit("Frozen test already evaluated here; inspect the existing report")
    dataset = build_topic_dataset()
    names = sorted(set(dataset["train"]["topic"]))
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=False)),
        ("classifier", LinearSVC(C=1.0, random_state=42, dual="auto", max_iter=5000)),
    ])
    start = time.perf_counter()
    model.fit(dataset["train"]["text"], dataset["train"]["label"])
    seconds = time.perf_counter() - start
    validation = topic_metrics(dataset["validation"]["label"], model.predict(dataset["validation"]["text"]), len(names))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "model.joblib")
    report = {"model": "TF-IDF word 1-2 grams + LinearSVC(C=1)", "seed": 42,
              "preprocessing_version": PREPROC_VERSION, "data_sha256": sha256(TOPIC_DATA),
              "labels": names, "split": split_evidence(dataset),
              "train_seconds": seconds, "validation": validation, "frozen_test": None}
    if args.evaluate_test:
        report["frozen_test"] = topic_metrics(dataset["test"]["label"], model.predict(dataset["test"]["text"]), len(names))
        write_json(test_path, report["frozen_test"])
    write_json(args.output_dir / "metrics.json", report)
    write_json(EVIDENCE / "tfidf_baseline.json", report)
    print(f"Validation macro-F1: {validation['macro_f1']:.6f}; fit time: {seconds:.2f}s")
    print("Citizen overlap:", report["split"]["citizen_overlap"])
    print("Frozen test:", report["frozen_test"] if args.evaluate_test else "not evaluated")


if __name__ == "__main__":
    main()
