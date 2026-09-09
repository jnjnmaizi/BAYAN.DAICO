"""Evaluate the supplied QA smoke set and a separately declared 9+3 diagnostic.

The updated course README requires 12 answerable rows and zero nulls. Preserve
the supplied file verbatim and select a supplemental 9-answer/3-null set
from distinct source contexts before running inference. Calibrate the no-answer
threshold on other contexts only. No QA weights are trained in this lab.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from bayan.models.qa import best_span
from bayan.models.training import EVIDENCE, ROOT, checkpoint_source, sha256, write_json


def read_squad(path):
    examples = []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for item in data["data"]:
        for paragraph in item["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                if bool(qa["is_impossible"]) != (len(qa["answers"]) == 0):
                    raise ValueError(f"Inconsistent answerability in {qa['id']}")
                for answer in qa["answers"]:
                    start = answer["answer_start"]
                    if context[start:start+len(answer["text"])] != answer["text"]:
                        raise ValueError(f"Bad answer offsets in {qa['id']}")
                examples.append({**qa, "context": context, "title": item["title"]})
    return examples


def protocol(source, smoke):
    """Fix development/supplemental membership independently of model outputs."""
    smoke_contexts = {row["context"] for row in smoke}
    by_context = defaultdict(list)
    seen = set()
    for row in source:
        key = (row["context"], row["question"])
        if row["context"] not in smoke_contexts and key not in seen:
            by_context[row["context"]].append(row)
            seen.add(key)
    supplemental, chosen = [], set()
    for context, rows in by_context.items():
        answerable = [row for row in rows if not row["is_impossible"]]
        nulls = [row for row in rows if row["is_impossible"]]
        if len(answerable) >= 3 and nulls:
            supplemental.extend(answerable[:3] + nulls[:1])
            chosen.add(context)
        if len(chosen) == 3:
            break
    if len(supplemental) != 12:
        raise ValueError("Cannot construct the predeclared supplemental 9+3 set")
    development = [row for context, rows in by_context.items() if context not in chosen for row in rows]
    assert not ({r['context'] for r in development} & (chosen | smoke_contexts))
    return development, supplemental


def infer(model, tokenizer, row):
    # Preserve original context: preprocessing here would invalidate gold offsets.
    encoded = tokenizer(row["question"], row["context"], return_tensors="pt", return_offsets_mapping=True)
    if encoded.input_ids.shape[1] > model.config.max_position_embeddings - 2:
        raise ValueError("Context is too long; a sliding-window implementation is required")
    offsets = encoded.pop("offset_mapping")[0].tolist()
    offsets = [tuple(offset) if seq == 1 else None for offset, seq in zip(offsets, encoded.sequence_ids(0))]
    with torch.inference_mode():
        output = model(**encoded)
    start = output.start_logits[0].numpy()
    end = output.end_logits[0].numpy()
    cls_index = encoded.input_ids[0].tolist().index(tokenizer.cls_token_id)
    null_score = float(start[cls_index] + end[cls_index])
    candidate = best_span(start, end, offsets, null_score=null_score, null_threshold=1e30)
    return {"id": row["id"], "candidate": candidate["answer"],
            "candidate_text": row["context"][slice(*candidate["answer"])] if candidate["answer"] else None,
            "score_diff": candidate["score_diff"]}


def score(rows, inferred, threshold):
    answered = nulls = answered_correct = null_correct = 0
    predictions = []
    for row, raw in zip(rows, inferred):
        answer = None if raw["score_diff"] is None or raw["score_diff"] > threshold else raw["candidate"]
        expected = [(a["answer_start"], a["answer_start"] + len(a["text"])) for a in row["answers"]]
        correct = answer is None if row["is_impossible"] else answer in expected
        if row["is_impossible"]:
            nulls += 1
            null_correct += int(correct)
        else:
            answered += 1
            answered_correct += int(correct)
        predictions.append({**raw, "answer": row["context"][slice(*answer)] if answer else None,
                            "answer_offsets": answer, "is_impossible": row["is_impossible"], "correct": correct})
    return {"answerable": answered, "answerable_exact_span_correct": answered_correct,
            "unanswerable": nulls, "null_correct": null_correct,
            "total": len(rows), "correct": answered_correct + null_correct,
            "predictions": predictions}


def course_smoke_assessment(supplied):
    """Assess the revised README target without changing predictions or thresholds."""
    matches = supplied["answerable"] == 12 and supplied["unanswerable"] == 0
    return {
        "source": "https://github.com/AljawharaAlbahlalDev/SDA-AIE-211-Bayan-Course/blob/7949de02de71cd3ae644cd89f766dfc73aa9b7f4/README.md#lab-3b--step-4-qa-smoke-set",
        "expected_answerable": 12,
        "expected_unanswerable": 0,
        "composition_matches": matches,
        "target_met": matches and supplied["answerable_exact_span_correct"] == 12,
        "null_handling_tested_by_supplied_set": supplied["unanswerable"] > 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="deepset/roberta-base-squad2")
    parser.add_argument("--revision", default="adc3b06f79f797d1c575d5479d6f5efe54a9e3b4")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/qa")
    args = parser.parse_args()
    report_path = args.output_dir / "metrics.json"
    if report_path.exists():
        raise SystemExit(f"QA evaluation already recorded: {report_path}")
    source_path = ROOT / "data/models/bayan_qa.json"
    smoke_path = ROOT / "data/eval/qa_smoke_set.json"
    smoke = read_squad(smoke_path)
    development, supplemental = protocol(read_squad(source_path), smoke)
    selection = {
        "development_ids": [r["id"] for r in development], "supplemental_ids": [r["id"] for r in supplemental],
        "supplied_ids": [r["id"] for r in smoke], "context_overlap": 0,
        "selection": "First three non-smoke source contexts with 3 unique answerable questions and a null; all other non-smoke contexts for calibration",
    }
    write_json(args.output_dir / "protocol.json", selection)
    write_json(EVIDENCE / "qa_protocol.json", selection)
    print(f"Supplied smoke: {len(smoke)} rows / {sum(r['is_impossible'] for r in smoke)} nulls; supplemental: 9+3; development: {len(development)}", flush=True)
    torch.set_num_threads(4)
    source = checkpoint_source(args.checkpoint, args.revision)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained(source, local_files_only=True, attn_implementation="eager").eval()
    start = time.perf_counter()
    dev_predictions = [infer(model, tokenizer, row) for row in development]
    differences = sorted({p["score_diff"] for p in dev_predictions if p["score_diff"] is not None})
    thresholds = [0.0] + ([differences[0] - 1] + differences if differences else [])

    def calibration_objective(threshold):
        result = score(development, dev_predictions, threshold)
        balanced = .5 * (result["answerable_exact_span_correct"] / result["answerable"] + result["null_correct"] / result["unanswerable"])
        return balanced, -abs(threshold), -threshold

    threshold = max(thresholds, key=calibration_objective)
    print(f"Threshold fixed using development only: {threshold:.4f}", flush=True)
    results = {}
    for name, rows in [("supplied_smoke", smoke), ("supplemental_9_plus_3", supplemental)]:
        results[name] = score(rows, [infer(model, tokenizer, row) for row in rows], threshold)
    dev = score(development, dev_predictions, threshold)
    supplied = results["supplied_smoke"]
    matches_contract = supplied["answerable"] == 9 and supplied["unanswerable"] == 3
    assessment = course_smoke_assessment(supplied)
    unique_questions = len({(r["context"], r["question"]) for r in smoke})
    report = {"checkpoint": args.checkpoint, "revision": Path(source).name, "device": "cpu",
              "method": "Pretrained SQuAD2 QA; no Bayan fine-tuning; strict character-span exact match",
              "source_sha256": sha256(source_path), "smoke_sha256": sha256(smoke_path),
              "null_rule": "null_score - best_span_score > threshold", "null_threshold": threshold,
              "development": {k:v for k,v in dev.items() if k != "predictions"},
              "seconds": time.perf_counter()-start, **results,
              "course_smoke_assessment": assessment,
              "supplied_9_plus_3_contract_met": matches_contract and supplied["correct"] == 12,
              "data_issue": None if assessment["composition_matches"] else (
                  f"Supplied smoke has {supplied['answerable']} answerable rows, {unique_questions} unique questions, "
                  f"{supplied['unanswerable']} nulls; expected 12 answerable rows and zero nulls per the updated README."),
              "limitations": "Supplemental 9+3 is separate evidence, not a replacement of the supplied test; contexts share synthetic templates."}
    write_json(report_path, report)
    write_json(EVIDENCE / "qa.json", report)
    print("Course smoke assessment:", assessment, flush=True)
    for name, result in results.items():
        print(name, {k:v for k,v in result.items() if k != "predictions"}, flush=True)


if __name__ == "__main__":
    main()
