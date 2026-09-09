"""Audit supplied Arabic dialect metadata, without claiming dialect prediction."""
import argparse
from collections import Counter
import csv
from pathlib import Path

from bayan.models.training import ROOT, sha256, write_json


def audit(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    arabic = [r for r in rows if r["lang"] == "ar"]
    if not arabic or any(not r["dialect_region"].strip() for r in arabic):
        raise ValueError("Expected Arabic rows with nonempty dialect metadata")
    counts = Counter(r["dialect_region"] for r in arabic)
    return {"data_sha256": sha256(path), "total_rows": len(rows), "arabic_rows": len(arabic),
            "method": "Count supplied dialect_region labels; no automatic dialect identification",
            "counts": dict(sorted(counts.items())),
            "percent_of_arabic": {k:100*v/len(arabic) for k,v in sorted(counts.items())},
            "by_split": {split:dict(Counter(r["dialect_region"] for r in arabic if r["split"] == split))
                         for split in ("train", "validation", "test")},
            "gulf_validation_available": any(r["dialect_region"] == "Gulf" and r["split"] == "validation" for r in arabic),
            "gulf_test_available": any(r["dialect_region"] == "Gulf" and r["split"] == "test" for r in arabic),
            "implication": "MSA-only evaluation excludes the Gulf majority; evaluate Gulf and MSA separately. Metadata is synthetic and does not prove linguistic dialect coverage."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/bayan_feedback.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/lab4/dialect_audit.json")
    args = parser.parse_args()
    report = audit(args.data)
    write_json(args.output, report)
    print(f"Arabic rows: {report['arabic_rows']} / {report['total_rows']}")
    for name, count in report["counts"].items():
        print(f"{name}: {count} ({report['percent_of_arabic'][name]:.2f}%)")
    print(report["implication"])


if __name__ == "__main__":
    main()
