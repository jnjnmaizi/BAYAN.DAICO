"""Optional CAMeLBERT Mix/DA validation comparison; no frozen-test access."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

from bayan.models.data import TOPIC_DATA, build_topic_dataset
from bayan.models.training import CACHE, ROOT, sha256, topic_metrics, write_json
from bayan.preprocessing.arabic import CAMELBERT_V1, prepare_arabic_text

CANDIDATES = {
    "mix": ("CAMeL-Lab/bert-base-arabic-camelbert-mix", "9be352797bdf28a9ae21e2ae582aaaca7abdb22d"),
    "da": ("CAMeL-Lab/bert-base-arabic-camelbert-da", "231698eab9ebf0ae7b518a64277b81b2fe829f2d"),
}


class ContiguousTrainer(Trainer):
    """Safetensors requires contiguous storage; MPS BERT weights may be strided."""
    def _save(self, output_dir=None, state_dict=None):
        state = self.model.state_dict() if state_dict is None else state_dict
        packed = {name:value.detach().cpu().contiguous() for name,value in state.items()}
        super()._save(output_dir=output_dir, state_dict=packed)


def slice_metrics(rows, predicted, num_labels):
    result = {}
    for name in ("all_arabic", "Gulf", "MSA"):
        indices = [i for i,r in enumerate(rows) if name == "all_arabic" or r["dialect_region"] == name]
        result[name] = topic_metrics([rows[i]["label"] for i in indices],
                                     [predicted[i] for i in indices], num_labels) if indices else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="Run the optional two-model training experiment")
    parser.add_argument("--model-root", type=Path, default=ROOT / "artifacts/lab4_models")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/lab4")
    parser.add_argument("--incumbent-dir", type=Path, default=ROOT / "artifacts/topic_classifier")
    args = parser.parse_args()
    path = args.output_dir / "arabic_bakeoff.json"
    if path.exists():
        print(path.read_text())
        return
    if not args.train:
        print("Optional experiment has no saved report. Run with --train to train Mix and DA on Arabic training data.")
        return
    torch.set_num_threads(4)
    dataset = build_topic_dataset()
    names = sorted(set(dataset["train"]["topic"]))
    label2id = {name:i for i,name in enumerate(names)}
    train = dataset["train"].filter(lambda r:r["lang"] == "ar")
    validation = dataset["validation"].filter(lambda r:r["lang"] == "ar")
    rows = list(validation)
    incumbent_meta = json.loads((args.incumbent_dir / "metrics.json").read_text())
    if incumbent_meta["data_sha256"] != sha256(TOPIC_DATA) or incumbent_meta["labels"] != names:
        raise ValueError("Incumbent and candidate data/labels differ")
    incumbent_predictions = {r["feedback_id"]:r for r in json.loads((args.incumbent_dir / "validation_predictions.json").read_text())}
    if any(incumbent_predictions[r["feedback_id"]]["label"] != r["topic"] for r in rows):
        raise ValueError("Incumbent validation gold labels differ")
    incumbent = slice_metrics(rows, [label2id[incumbent_predictions[r["feedback_id"]]["prediction"]] for r in rows], len(names))
    protocol = {"data_sha256":sha256(TOPIC_DATA), "split":"supplied training + validation, Arabic only",
                "train_rows":len(train), "validation_rows":len(validation), "seed":42,
                "profile":"camelbert_v1, dediacritize=True after Lab 1 preprocessing",
                "epochs":2, "learning_rate":2e-5, "batch_size":16, "gradient_accumulation":2,
                "max_length":64, "candidates":CANDIDATES, "frozen_test_used":False,
                "selection":"Highest Gulf validation macro-F1, then all-Arabic macro-F1; retain incumbent on a tie",
                "incumbent_predictions_sha256":sha256(args.incumbent_dir / "validation_predictions.json")}
    write_json(args.output_dir / "arabic_bakeoff_protocol.json", protocol)
    results = {}
    for name, (checkpoint, revision) in CANDIDATES.items():
        directory = args.model_root / name
        report_path = directory / "metrics.json"
        if report_path.exists():
            saved = json.loads(report_path.read_text())
            if saved["protocol"] != json.loads(json.dumps(protocol)):
                raise ValueError("Saved run protocol differs; preserve it and use a separate experiment")
            results[name] = saved
            continue
        set_seed(42)
        source = snapshot_download(checkpoint, revision=revision, cache_dir=CACHE,
                                   allow_patterns=["config.json", "pytorch_model.bin", "vocab.txt",
                                                   "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"])
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            source, num_labels=len(names), id2label=dict(enumerate(names)), label2id=label2id,
            local_files_only=True, attn_implementation="eager")

        def tokenize(batch):
            text = [prepare_arabic_text(t, CAMELBERT_V1)["model_text"] for t in batch["text"]]
            return tokenizer(text, truncation=True, max_length=64)

        encoded = {split:ds.map(tokenize, batched=True, remove_columns=[c for c in ds.column_names if c != "label"])
                   for split,ds in [("train",train),("validation",validation)]}
        training_args = TrainingArguments(
            output_dir=str(directory / "checkpoints"), num_train_epochs=2,
            per_device_train_batch_size=16, per_device_eval_batch_size=16, gradient_accumulation_steps=2,
            learning_rate=2e-5, weight_decay=.01, warmup_ratio=.1, evaluation_strategy="epoch",
            save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
            metric_for_best_model="macro_f1", greater_is_better=True, group_by_length=True,
            logging_steps=50, report_to=[], seed=42, data_seed=42, optim="adamw_torch",
            dataloader_pin_memory=False, disable_tqdm=True)
        trainer = ContiguousTrainer(model=model, args=training_args, tokenizer=tokenizer,
                          data_collator=DataCollatorWithPadding(tokenizer),
                          train_dataset=encoded["train"], eval_dataset=encoded["validation"],
                          compute_metrics=lambda p:topic_metrics(p.label_ids, np.argmax(p.predictions,axis=-1),len(names)))
        print(f"Training {name}: {training_args.device}", flush=True)
        started = time.perf_counter()
        trainer.train()
        trainer.save_model(str(directory))
        tokenizer.save_pretrained(directory)
        predictions = trainer.predict(encoded["validation"]).predictions.argmax(-1).tolist()
        result = {"checkpoint":checkpoint, "revision":revision, "protocol":protocol,
                  "device":str(training_args.device), "seconds":time.perf_counter()-started,
                  "slices":slice_metrics(rows,predictions,len(names)), "history":trainer.state.log_history}
        write_json(report_path,result)
        write_json(directory / "validation_predictions.json", [{"feedback_id":r["feedback_id"],
                   "gold":r["topic"],"prediction":names[p]} for r,p in zip(rows,predictions)])
        write_json(args.output_dir / f"camelbert_{name}.json",result)
        results[name] = result
        del trainer, model, tokenizer, encoded
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    ranked = {"xlm-roberta-base":incumbent, **{name:r["slices"] for name,r in results.items()}}
    gulf_available = all(slices["Gulf"] is not None for slices in ranked.values())
    winner = max(ranked, key=lambda name:(ranked[name]["Gulf"]["macro_f1"], ranked[name]["all_arabic"]["macro_f1"])) if gulf_available else "xlm-roberta-base"
    report = {"protocol":protocol,"incumbent":incumbent,"candidates":results,"selected":winner,
              "selection_status":"measured Gulf comparison" if gulf_available else "Gulf validation absent; incumbent retained, no Gulf winner established",
              "da_vs_incumbent_gulf_delta_points":100*(ranked["da"]["Gulf"]["macro_f1"]-incumbent["Gulf"]["macro_f1"]) if gulf_available else None,
              "da_vs_mix_gulf_delta_points":100*(ranked["da"]["Gulf"]["macro_f1"]-ranked["mix"]["Gulf"]["macro_f1"]) if gulf_available else None,
              "limitations":"Validation comparison for model selection, not frozen-test evidence. Candidates use Arabic-only training and model-specific preprocessing; comparison does not isolate pretraining dialect alone. Synthetic text repetition limits generalization."}
    report["plus_4_point_target_met"] = report["da_vs_incumbent_gulf_delta_points"] >= 4 if gulf_available else None
    write_json(path,report)
    print(json.dumps({k:report[k] for k in ["selected","da_vs_incumbent_gulf_delta_points","da_vs_mix_gulf_delta_points","plus_4_point_target_met"]},indent=2),flush=True)


if __name__ == "__main__":
    main()
