"""Check whether the +8-point target is attainable, using saved evidence only."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def target_feasibility(class_support, baseline_macro_f1, required_delta_points=8):
    """Fixed-label macro-F1, zero_division=0: absent gold classes have F1=0."""
    if not class_support or any(count < 0 for count in class_support.values()):
        raise ValueError("Expected nonnegative class counts")
    ceiling = sum(count > 0 for count in class_support.values()) / len(class_support)
    if not 0 <= baseline_macro_f1 <= ceiling + 1e-12:
        raise ValueError("Baseline is incompatible with the declared metric and class support")
    headroom = 100 * (ceiling - baseline_macro_f1)
    return {
        "metric_policy": "macro-F1 over all training labels; zero_division=0",
        "class_support": dict(class_support),
        "missing_classes": [name for name, count in class_support.items() if count == 0],
        "maximum_possible_macro_f1": ceiling,
        "baseline_macro_f1": baseline_macro_f1,
        "maximum_possible_improvement_points": headroom,
        "required_improvement_points": required_delta_points,
        "target_attainable_on_this_frozen_test": headroom + 1e-9 >= required_delta_points,
    }


def build_report(data_path, baseline_path, classifier_path):
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    baseline = json.loads(baseline_path.read_text())
    classifier = json.loads(classifier_path.read_text())
    if any(report["data_sha256"] != digest for report in (baseline, classifier)):
        raise ValueError("Saved metrics refer to different data; cannot reuse them")
    with data_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = sorted({r["topic"] for r in rows if r["split"] == "train"})
    if any(report["labels"] != labels for report in (baseline, classifier)):
        raise ValueError("Saved label vocabulary does not match training data")
    test = [row for row in rows if row["split"] == "test"]
    counts = Counter(row["topic"] for row in test)
    if not test or set(counts) - set(labels):
        raise ValueError("Expected a nonempty test with known labels")
    for report in (baseline, classifier):
        metric = report["frozen_test"]
        if metric["n"] != len(test) or metric["num_task_labels"] != len(labels):
            raise ValueError("Saved metric shape does not match this test")
    result = target_feasibility({label: counts[label] for label in labels}, baseline["frozen_test"]["macro_f1"])
    result.update({
        "data_sha256": digest,
        "baseline_evidence_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "classifier_evidence_sha256": hashlib.sha256(classifier_path.read_bytes()).hexdigest(),
        "test_rows": len(test),
        "baseline_accuracy": baseline["frozen_test"]["accuracy"],
        "classifier_macro_f1": classifier["frozen_test"]["macro_f1"],
        "classifier_accuracy": classifier["frozen_test"]["accuracy"],
        "observed_delta_points": 100 * (classifier["frozen_test"]["macro_f1"] - baseline["frozen_test"]["macro_f1"]),
        "new_training_or_test_inference": False,
        "next_step": "Use a representative unseen evaluation set and a predeclared comparison protocol to measure performance beyond this ceiling. Do not weaken the baseline or tune on this test.",
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/bayan_feedback.csv")
    parser.add_argument("--baseline", type=Path, default=ROOT / "artifacts/lab3/tfidf_baseline.json")
    parser.add_argument("--classifier", type=Path, default=ROOT / "artifacts/lab3/topic_classifier.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/lab3/topic_target_audit.json")
    args = parser.parse_args()
    report = build_report(args.data, args.baseline, args.classifier)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
