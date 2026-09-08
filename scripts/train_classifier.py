"""Fine-tune XLM-R on training citizens; select epochs on validation only."""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import transformers
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

from bayan.models.data import TOPIC_DATA, build_topic_dataset
from bayan.models.training import (EVIDENCE, ROOT, checkpoint_source, predictions,
                                   sha256, split_evidence, topic_metrics, write_json)
from bayan.preprocessing.core import PREPROC_VERSION


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts/topic_classifier"),
        help="Where to save the trained classifier artefact (local path or mounted Drive path).",
    )
    parser.add_argument("--checkpoint", default="xlm-roberta-base")
    parser.add_argument("--revision", default="e73636d4f797dec63c3081bb6ed5c7b0bb3f2089")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=-1, help="Optional bounded engineering smoke run; use a separate output directory")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true", help="One-shot final evaluation, including the saved baseline")
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "artifacts/tfidf_baseline")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "metrics.json"
    if (output_dir / "frozen_test.json").exists():
        raise SystemExit("This artifact has already seen the frozen test; keep its recorded results")
    if report_path.exists() and not args.evaluate_only:
        raise SystemExit("Training output already exists; use --evaluate-only or a new output directory")
    if args.evaluate_test and not args.evaluate_only:
        raise SystemExit("Train/select on validation first, then use --evaluate-only --evaluate-test")
    torch.set_num_threads(4)
    set_seed(42)
    dataset = build_topic_dataset()
    names = sorted(set(dataset["train"]["topic"]))
    id2label = dict(enumerate(names))
    label2id = {name: i for i, name in id2label.items()}
    if args.evaluate_only:
        report = json.loads(report_path.read_text())
        if report["data_sha256"] != sha256(TOPIC_DATA) or report["preprocessing_version"] != PREPROC_VERSION:
            raise ValueError("Data/preprocessing changed since training")
        args.max_length = report["max_length"]
        source = str(output_dir)
    else:
        source = checkpoint_source(args.checkpoint, args.revision)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=len(names), id2label=id2label, label2id=label2id,
        local_files_only=True, attn_implementation="eager",
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    needed = ["validation"] if args.evaluate_only else ["train", "validation"]
    if args.evaluate_test:
        needed.append("test")
    encoded = {split: dataset[split].map(tokenize, batched=True, desc=f"Tokenize {split}",
                                        remove_columns=[c for c in dataset[split].column_names if c != "label"])
               for split in needed}
    train_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, weight_decay=0.01, warmup_ratio=0.1,
        evaluation_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        group_by_length=True, logging_steps=25, report_to=[], seed=42, data_seed=42,
        optim="adamw_torch", use_cpu=args.cpu, dataloader_pin_memory=False,
        max_steps=args.max_steps, disable_tqdm=True,
    )
    trainer = Trainer(model=model, args=train_args, tokenizer=tokenizer,
                      data_collator=DataCollatorWithPadding(tokenizer),
                      train_dataset=encoded.get("train"), eval_dataset=encoded["validation"],
                      compute_metrics=lambda p: topic_metrics(p.label_ids, np.argmax(p.predictions, axis=-1), len(names)))
    print(f"Device: {trainer.args.device}; parameters: {model.num_parameters():,}; batch: {args.batch_size}; max length: {args.max_length}", flush=True)
    if not args.evaluate_only:
        start = time.perf_counter()
        trainer.train()
        seconds = time.perf_counter() - start
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(output_dir)
        validation = trainer.predict(encoded["validation"])
        report = {"checkpoint": args.checkpoint, "revision": Path(source).name,
                  "preprocessing_version": PREPROC_VERSION, "data_sha256": sha256(TOPIC_DATA),
                  "labels": names, "split": split_evidence(dataset), "seed": 42,
                  "epochs_requested": args.epochs, "max_steps": args.max_steps,
                  "global_steps": trainer.state.global_step, "best_checkpoint": trainer.state.best_model_checkpoint,
                  "learning_rate": args.learning_rate, "max_length": args.max_length,
                  "batch_size": args.batch_size, "gradient_accumulation": args.gradient_accumulation,
                  "device": str(trainer.args.device), "torch_version": torch.__version__,
                  "transformers_version": transformers.__version__, "train_seconds": seconds,
                  "validation": topic_metrics(validation.label_ids, validation.predictions.argmax(-1), len(names)),
                  "history": trainer.state.log_history, "frozen_test": None}
        write_json(output_dir / "validation_predictions.json", predictions(dataset["validation"], validation.predictions, id2label))
    if args.evaluate_test:
        if report["max_steps"] > 0:
            raise ValueError("Bounded smoke runs must not be evaluated on the frozen test")
        baseline_meta = json.loads((args.baseline_dir / "metrics.json").read_text())
        if baseline_meta["data_sha256"] != report["data_sha256"] or baseline_meta["labels"] != names or baseline_meta["preprocessing_version"] != PREPROC_VERSION:
            raise ValueError("Baseline and transformer must use identical data, labels and preprocessing")
        if (args.baseline_dir / "frozen_test.json").exists():
            baseline_result = json.loads((args.baseline_dir / "frozen_test.json").read_text())
        else:
            baseline = joblib.load(args.baseline_dir / "model.joblib")
            baseline_result = topic_metrics(dataset["test"]["label"], baseline.predict(dataset["test"]["text"]), len(names))
            write_json(args.baseline_dir / "frozen_test.json", baseline_result)
        result = trainer.predict(encoded["test"])
        report["frozen_test"] = topic_metrics(result.label_ids, result.predictions.argmax(-1), len(names))
        report["baseline_frozen_test"] = baseline_result
        report["delta_macro_f1_points"] = 100 * (report["frozen_test"]["macro_f1"] - baseline_result["macro_f1"])
        report["target_met"] = report["delta_macro_f1_points"] >= 8
        write_json(output_dir / "frozen_test.json", report["frozen_test"])
        baseline_meta["frozen_test"] = baseline_result
        write_json(args.baseline_dir / "metrics.json", baseline_meta)
        write_json(EVIDENCE / "tfidf_baseline.json", baseline_meta)
        write_json(output_dir / "test_predictions.json", predictions(dataset["test"], result.predictions, id2label))
    write_json(report_path, report)
    if report["max_steps"] < 0:
        write_json(EVIDENCE / "topic_classifier.json", report)
    print(json.dumps({k: report[k] for k in ["validation", "frozen_test", "train_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
