"""Count base-encoder parameters from checkpoint configs; no weights needed."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel

CHECKPOINTS = {
    "bert-base-multilingual-cased": "3f076fdb1ab68d5b2880cb87a0886f315b8146f8",
    "CAMeL-Lab/bert-base-arabic-camelbert-mix": "9be352797bdf28a9ae21e2ae582aaaca7abdb22d",
}
ROOT = Path(__file__).resolve().parents[1]


def count_parameters(model) -> dict:
    """Disjoint buckets: all LayerNorms (including embedding norms) go to norms."""
    buckets = dict.fromkeys(("embeddings", "attention", "ffn", "norms", "pooler", "other"), 0)
    for name, parameter in model.named_parameters():
        if "LayerNorm" in name:
            bucket = "norms"
        elif name.startswith("embeddings."):
            bucket = "embeddings"
        elif ".attention." in name:
            bucket = "attention"
        elif name.startswith("encoder.layer.") and (
            ".intermediate.dense." in name or ".output.dense." in name
        ):
            bucket = "ffn"
        elif name.startswith("pooler."):
            bucket = "pooler"
        else:
            bucket = "other"
        buckets[bucket] += parameter.numel()
    total = sum(parameter.numel() for parameter in model.parameters())
    assert sum(buckets.values()) == total
    return {"total_params": total, "buckets": buckets,
            "embeddings_pct": 100 * buckets["embeddings"] / total}


def audit(checkpoint: str) -> dict:
    revision = CHECKPOINTS.get(checkpoint, "main")
    # Prefer the existing Lab 1 cache without network requests or cache writes.
    try:
        config = AutoConfig.from_pretrained(checkpoint, revision=revision, local_files_only=True)
    except OSError:
        config = AutoConfig.from_pretrained(
            checkpoint, revision=revision, cache_dir=ROOT / "artifacts/hf_cache"
        )
    if config.model_type != "bert":
        raise ValueError("This audit's parameter buckets are defined for BERT encoders")
    # Meta tensors retain exact parameter shapes without allocating model weights.
    with torch.device("meta"):
        model = AutoModel.from_config(config)
    return {"checkpoint": checkpoint, "revision": config._commit_hash,
            "scope": "BertModel encoder + pooler; no MLM/task head; all LayerNorms in norms",
            "method": "named_parameters on a meta-device model built from checkpoint config",
            "vocab_size": config.vocab_size, "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers, **count_parameters(model)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/lab2")
    args = parser.parse_args()
    results = [audit(checkpoint) for checkpoint in CHECKPOINTS]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "parameter_audit.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print("Base encoder + pooler; all LayerNorms in norms; no MLM/task head")
    print("| Checkpoint | Vocabulary | Total | Embeddings | Embeddings % | Attention | FFN | Norms | Pooler | Other |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        b = result["buckets"]
        print(f"| {result['checkpoint']} | {result['vocab_size']:,} | {result['total_params']:,} | "
              f"{b['embeddings']:,} | {result['embeddings_pct']:.2f}% | {b['attention']:,} | "
              f"{b['ffn']:,} | {b['norms']:,} | {b['pooler']:,} | {b['other']:,} |")


if __name__ == "__main__":
    main()
