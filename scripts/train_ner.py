"""Fine-tune XLM-R NER with first-piece labels and template-disjoint splits."""
import argparse
from collections import Counter
import json
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import Dataset
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer, TrainingArguments, set_seed)

from bayan.models.ner import TAGS, align_labels, read_conll, split_ner
from bayan.models.training import EVIDENCE, ROOT, checkpoint_source, sha256, write_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts/ner"),
        help="Where to save the trained NER artefact (local path or mounted Drive path).",
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/bayan_ner.conll")
    parser.add_argument("--checkpoint", default="xlm-roberta-base")
    parser.add_argument("--revision", default="e73636d4f797dec63c3081bb6ed5c7b0bb3f2089")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    return parser.parse_args()


def decode(predictions, labels):
    predicted = np.argmax(predictions, axis=-1)
    gold, guesses = [], []
    for pred, true in zip(predicted, labels):
        gold.append([TAGS[int(t)] for t in true if t != -100])
        guesses.append([TAGS[int(p)] for p, t in zip(pred, true) if t != -100])
    return gold, guesses


def metrics(prediction):
    gold, guesses = decode(prediction.predictions, prediction.label_ids)
    options = dict(mode="strict", scheme=IOB2, zero_division=0)
    return {"entity_f1": float(f1_score(gold, guesses, **options)),
            "entity_precision": float(precision_score(gold, guesses, **options)),
            "entity_recall": float(recall_score(gold, guesses, **options))}


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "metrics.json"
    if (output_dir / "frozen_test.json").exists():
        raise SystemExit("This artifact already has a frozen-test result")
    if report_path.exists() and not args.evaluate_only:
        raise SystemExit("Training output exists; use --evaluate-only or a new directory")
    torch.set_num_threads(4)
    set_seed(42)
    records = read_conll(args.data)
    splits = split_ner(records)
    if args.evaluate_only:
        report = json.loads(report_path.read_text())
        if report["data_sha256"] != sha256(args.data):
            raise ValueError("Data changed since training")
        args.max_length = report["max_length"]
        source = str(output_dir)
    else:
        source = checkpoint_source(args.checkpoint, args.revision)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(
        source, num_labels=len(TAGS), id2label=dict(enumerate(TAGS)),
        label2id={tag: i for i, tag in enumerate(TAGS)}, local_files_only=True, attn_implementation="eager",
    )
    tag2id = {tag: i for i, tag in enumerate(TAGS)}

    def tokenize(batch):
        encoded = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True, max_length=args.max_length)
        encoded["labels"] = [align_labels(encoded.word_ids(i), [tag2id[t] for t in tags]) for i, tags in enumerate(batch["tags"])]
        # Truncated gold entities would inflate or invalidate entity-level scores.
        for i, tags in enumerate(batch["tags"]):
            if set(w for w in encoded.word_ids(i) if w is not None) != set(range(len(tags))):
                raise ValueError("NER truncation would drop labelled words; increase --max-length")
        return encoded

    needed = ["validation"] if args.evaluate_only else ["train", "validation"]
    if args.evaluate_test:
        needed.append("test")
    datasets = {}
    for split in needed:
        ds = Dataset.from_list(splits[split])
        datasets[split] = ds.map(tokenize, batched=True, remove_columns=ds.column_names, desc=f"NER {split}")
    train_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2, learning_rate=args.learning_rate, weight_decay=.01,
        warmup_ratio=.1, evaluation_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="entity_f1", greater_is_better=True,
        group_by_length=True, logging_steps=25, report_to=[], seed=42, data_seed=42,
        optim="adamw_torch", use_cpu=args.cpu, dataloader_pin_memory=False, disable_tqdm=True,
    )
    trainer = Trainer(model=model, args=train_args, tokenizer=tokenizer,
                      data_collator=DataCollatorForTokenClassification(tokenizer),
                      train_dataset=datasets.get("train"), eval_dataset=datasets["validation"], compute_metrics=metrics)
    print(f"NER device: {trainer.args.device}; rows: { {s:len(r) for s,r in splits.items()} }", flush=True)
    if not args.evaluate_only:
        start = time.perf_counter()
        trainer.train()
        seconds = time.perf_counter() - start
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(output_dir)
        result = trainer.predict(datasets["validation"])
        gold, guesses = decode(result.predictions, result.label_ids)
        report = {"checkpoint": args.checkpoint, "revision": Path(source).name, "data_sha256": sha256(args.data),
                  "seed": 42, "max_length": args.max_length, "epochs": args.epochs,
                  "batch_size": args.batch_size, "gradient_accumulation": 2, "learning_rate": args.learning_rate,
                  "device": str(trainer.args.device), "train_seconds": seconds,
                  "torch_version": torch.__version__, "transformers_version": transformers.__version__,
                  "rows": {s:len(r) for s,r in splits.items()},
                  "template_groups": {s:len({r['template_group'] for r in rows}) for s,rows in splits.items()},
                  "template_overlap": 0, "split_policy": "seeded 70/20/10 by template groups, not rows; reference values excluded from group key",
                  "source_tags": dict(Counter(t for r in records for t in r["tags"])),
                  "evaluation": "seqeval strict IOB2 entity-level; special and continuation pieces ignored",
                  "validation": metrics(result), "validation_entities": classification_report(gold, guesses, mode="strict", scheme=IOB2, zero_division=0, output_dict=True),
                  "history": trainer.state.log_history, "frozen_test": None,
                  "limitations": "Synthetic templates, fixed date, no ORGANISATION annotations; no evidence for unseen dates or organisations."}
        write_json(output_dir / "split_manifest.json", {s:[{"id":r["id"],"template_group":r["template_group"]} for r in rows] for s,rows in splits.items()})
        write_json(output_dir / "validation_predictions.json", [{"id":r["id"],"tokens":r["tokens"],"gold":g,"prediction":p} for r,g,p in zip(splits["validation"],gold,guesses)])
        # Persist training/validation metadata before the one-shot test step.
        write_json(report_path, report)
    if args.evaluate_test:
        result = trainer.predict(datasets["test"])
        gold, guesses = decode(result.predictions, result.label_ids)
        report["frozen_test"] = metrics(result)
        report["test_entities"] = classification_report(gold, guesses, mode="strict", scheme=IOB2, zero_division=0, output_dict=True)
        report["target_met"] = report["frozen_test"]["entity_f1"] >= .8
        write_json(output_dir / "frozen_test.json", report["frozen_test"])
        write_json(output_dir / "test_predictions.json", [{"id":r["id"],"tokens":r["tokens"],"gold":g,"prediction":p} for r,g,p in zip(splits["test"],gold,guesses)])
    write_json(report_path, report)
    write_json(EVIDENCE / "ner.json", report)
    print(json.dumps({k:report[k] for k in ["validation","frozen_test","train_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
