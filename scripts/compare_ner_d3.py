"""Compare saved validation predictions; no training or model inference."""
import argparse
import json
from pathlib import Path

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

from bayan.models.training import ROOT, sha256, write_json


def compare(baseline_dir, candidate_dir):
    before_meta = json.loads((baseline_dir / "metrics.json").read_text())
    after_meta = json.loads((candidate_dir / "metrics.json").read_text())
    if before_meta["data_sha256"] != after_meta["data_sha256"]:
        raise ValueError("Training data versions differ")
    if before_meta.get("segmentation", "none") != "none" or after_meta.get("segmentation") != "camel_d3":
        raise ValueError("Expected original and D3-trained NER artifacts")
    before_path = baseline_dir / "validation_predictions.json"
    after_path = candidate_dir / "validation_predictions.json"
    before = json.loads(before_path.read_text())
    after = json.loads(after_path.read_text())
    if not before or [(r["id"],r["tokens"],r["gold"]) for r in before] != [(r["id"],r["tokens"],r["gold"]) for r in after]:
        raise ValueError("Validation records, original words or gold labels differ")
    for row in before + after:
        if len(row["prediction"]) != len(row["gold"]):
            raise ValueError("Predictions must align with every original word")
    gold = [r["gold"] for r in before]
    options = dict(mode="strict", scheme=IOB2, zero_division=0, output_dict=True)
    baseline = classification_report(gold, [r["prediction"] for r in before], **options)
    candidate = classification_report(gold, [r["prediction"] for r in after], **options)
    delta = float(100*(candidate["LOCATION"]["recall"] - baseline["LOCATION"]["recall"]))
    return {
        "scope":"Paired validation comparison of original and D3-trained models; follow-up development experiment",
        "rows":len(before), "data_sha256":before_meta["data_sha256"],
        "baseline_predictions_sha256":sha256(before_path), "candidate_predictions_sha256":sha256(after_path),
        "baseline":baseline, "d3_trained":candidate,
        "location_recall_delta_points":delta,
        "plus_4_point_target_met":delta >= 4,
        "adopt_d3":bool(delta > 0 and candidate["micro avg"]["f1-score"] >= baseline["micro avg"]["f1-score"]),
        "frozen_test_used":False,
        "explanation":"A tie restores quality but is not a +4-point gain. Keep the original path absent improvement; validation was used for checkpoint selection, so this is not independent test evidence.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "artifacts/ner")
    parser.add_argument("--candidate-dir", type=Path, default=ROOT / "artifacts/ner_d3")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/lab4/ner_d3_comparison.json")
    args = parser.parse_args()
    report = compare(args.baseline_dir, args.candidate_dir)
    write_json(args.output, report)
    print(json.dumps({k:report[k] for k in ["rows","location_recall_delta_points","plus_4_point_target_met","adopt_d3"]}, indent=2))


if __name__ == "__main__":
    main()
