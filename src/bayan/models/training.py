"""Shared, small utilities for reproducible Lab 3 experiments."""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "artifacts/hf_cache"
EVIDENCE = ROOT / "artifacts/lab3"


def write_json(path, value):
    def native(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(f"Unsupported JSON value: {type(item).__name__}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=native) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def topic_metrics(labels, predicted, num_labels):
    support = np.bincount(np.asarray(labels, dtype=int), minlength=num_labels)
    return {"macro_f1": float(f1_score(labels, predicted, labels=list(range(num_labels)), average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(labels, predicted)), "n": len(labels),
            "num_labels_with_support": int(np.count_nonzero(support)),
            "num_task_labels": num_labels}


def checkpoint_source(checkpoint, revision=None):
    """Resolve a cached snapshot first; downloads use the project cache."""
    from huggingface_hub import snapshot_download
    if Path(checkpoint).is_dir():
        return str(Path(checkpoint).resolve())
    options = dict(repo_id=checkpoint, revision=revision, cache_dir=CACHE,
                   allow_patterns=["config.json", "tokenizer_config.json", "tokenizer.json",
                                   "special_tokens_map.json", "sentencepiece.bpe.model", "vocab.json",
                                   "vocab.txt", "merges.txt", "model.safetensors"])
    try:
        return snapshot_download(**options, local_files_only=True)
    except OSError:
        return snapshot_download(**options)


def split_evidence(dataset):
    from bayan.models.data import SPLITS
    groups = {s: set(dataset[s]["citizen_group_id"]) for s in SPLITS}
    return {"rows": {s: len(dataset[s]) for s in SPLITS},
            "citizen_groups": {s: len(groups[s]) for s in SPLITS},
            "citizen_overlap": {f"{a}/{b}": len(groups[a] & groups[b]) for a,b in
                                (("train","validation"),("train","test"),("validation","test"))},
            "split_policy": "supplied frozen 70/20/10, unchanged",
            "validation_text_overlap_with_train": len(set(dataset["validation"]["text"]) & set(dataset["train"]["text"]))}


def predictions(dataset, logits, id2label):
    predicted = np.asarray(logits).argmax(-1)
    return [{"feedback_id": row["feedback_id"], "citizen_group_id": row["citizen_group_id"],
             "lang": row["lang"], "dialect_region": row["dialect_region"],
             "label": row["topic"], "prediction": id2label[int(pred)]}
            for row, pred in zip(dataset, predicted)]
