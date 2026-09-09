"""Fixed-model D3 ablation on Lab 3 validation; never reopen the frozen test."""
import argparse
import json
from pathlib import Path
import time

import torch
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from transformers import AutoModelForTokenClassification, AutoTokenizer

from bayan.models.ner import TAGS, read_conll, split_ner
from bayan.models.training import ROOT, sha256, write_json
from bayan.preprocessing.arabic import ner_segmented_view, segment


def entity_metrics(gold, predicted):
    return classification_report(gold, predicted, mode="strict", scheme=IOB2,
                                 zero_division=0, output_dict=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/ner")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/lab4")
    args = parser.parse_args()
    result_path = args.output_dir / "ner_segmentation.json"
    if result_path.exists():
        print(result_path.read_text())
        return
    data_path = ROOT / "data/models/bayan_ner.conll"
    source_meta = json.loads((args.model_dir / "metrics.json").read_text())
    if source_meta["data_sha256"] != sha256(data_path) or source_meta.get("segmentation", "none") != "none":
        raise ValueError("Expected the saved Lab 3 unsegmented NER model and original data")
    rows = split_ner(read_conll(data_path))["validation"]
    baseline_path = args.model_dir / "validation_predictions.json"
    baseline = json.loads(baseline_path.read_text())
    if [(r["id"], r["tokens"], r["tags"]) for r in rows] != [(r["id"], r["tokens"], r["gold"]) for r in baseline]:
        raise ValueError("Baseline validation records differ; cannot make a paired comparison")
    protocol = {
        "split": "Lab 3 validation only", "ids": [r["id"] for r in rows],
        "data_sha256": sha256(data_path), "model_sha256": sha256(args.model_dir / "model.safetensors"),
        "baseline_predictions_sha256": sha256(baseline_path),
        "segmentation": "CAMeL MLE calima-msa-r13, d3tok, undiacritized",
        "projection": "First subword of sole lexical stem per original word; gold BIO and source word spans unchanged",
        "model_updated": False, "frozen_test_used": False,
        "decision_rule": "Keep original path unless validation LOCATION recall improves with no aggregate entity F1 loss; no parameter sweep",
    }
    write_json(args.output_dir / "ner_segmentation_protocol.json", protocol)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir, local_files_only=True,
                                                           attn_implementation="eager").eval()
    if [model.config.id2label[i] for i in range(len(TAGS))] != TAGS:
        raise ValueError("Saved model label vocabulary differs")
    guesses, examples = [], []
    started = time.perf_counter()
    for start in range(0, len(rows), 32):
        batch = rows[start:start+32]
        views = [ner_segmented_view(r["tokens"]) for r in batch]
        encoded = tokenizer([pieces for pieces, _ in views], is_split_into_words=True,
                            padding=True, truncation=False, return_tensors="pt")
        if encoded.input_ids.shape[1] > model.config.max_position_embeddings - 2:
            raise ValueError("Segmented sequence exceeds model positions")
        with torch.inference_mode():
            predicted = model(**encoded).logits.argmax(-1).tolist()
        for i, ((pieces, anchors), prediction) in enumerate(zip(views, predicted)):
            word_ids = encoded.word_ids(i)
            if {w for w in word_ids if w is not None} != set(range(len(pieces))):
                raise ValueError("Tokenizer dropped segmented words")
            result = [TAGS[prediction[word_ids.index(anchor)]] for anchor in anchors]
            guesses.append(result)
            if len(examples) < 5:
                examples.append({"id":batch[i]["id"], "original":batch[i]["tokens"],
                                 "segmented":pieces, "stem_positions":anchors,
                                 "gold":batch[i]["tags"], "prediction":result})
    gold = [r["tags"] for r in rows]
    before = entity_metrics(gold, [r["prediction"] for r in baseline])
    after = entity_metrics(gold, guesses)
    delta = float(100*(after["LOCATION"]["recall"] - before["LOCATION"]["recall"]))
    adopt = bool(delta > 0 and after["micro avg"]["f1-score"] >= before["micro avg"]["f1-score"])
    report = {"protocol": {k:v for k,v in protocol.items() if k != "ids"},
              "rows":len(rows), "device":"cpu", "seconds":time.perf_counter()-started,
              "baseline":before, "segmented":after, "location_recall_delta_points":delta,
              "maximum_possible_location_gain_points":100*(1-before["LOCATION"]["recall"]),
              "plus_4_point_target_met":delta >= 4, "adopt_segmentation_for_saved_model":adopt,
              "examples":examples, "course_example":segment("وبالرياض"),
              "limitations":"Fixed-model validation ablation, not a new held-out estimate. Saved model was trained on unsegmented words; changed inputs may reduce performance. Synthetic templates and MSA morphology limit dialect generalization."}
    write_json(result_path, report)
    print(json.dumps({k:report[k] for k in ["rows","seconds","location_recall_delta_points",
          "maximum_possible_location_gain_points","plus_4_point_target_met","adopt_segmentation_for_saved_model"]}, indent=2))


if __name__ == "__main__":
    main()
