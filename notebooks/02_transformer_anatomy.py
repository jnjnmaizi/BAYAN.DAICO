"""Run Lab 2 numerical checks and pretrained mBERT attention diagnostics on CPU.

Outputs: metrics JSON, an annotated pretrained attention map, and a causal map.
The first run downloads pinned model weights; subsequent runs use the cache.
"""
import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoTokenizer

from bayan.attention import MultiHeadAttention, attention
from bayan.preprocessing.core import normalize, mask_pii

CHECKPOINT = "bert-base-multilingual-cased"
REVISION = "3f076fdb1ab68d5b2880cb87a0886f315b8146f8"


def numerical_checks(output_dir):
    torch.manual_seed(42)
    q, k, v = (torch.randn(2, 4, 6, 8) for _ in range(3))
    actual = attention(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
    causal, weights = attention(q, k, v, is_causal=True, need_weights=True)
    torch.testing.assert_close(
        causal, F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-6, rtol=0
    )
    assert torch.count_nonzero(weights.triu(1)) == 0
    model = MultiHeadAttention(32, 4).eval()
    reference = torch.nn.MultiheadAttention(32, 4, dropout=0, batch_first=True).eval()
    with torch.no_grad():
        reference.in_proj_weight.copy_(torch.cat([model.q_proj.weight, model.k_proj.weight, model.v_proj.weight]))
        reference.in_proj_bias.copy_(torch.cat([model.q_proj.bias, model.k_proj.bias, model.v_proj.bias]))
        reference.out_proj.load_state_dict(model.out_proj.state_dict())
        x = torch.randn(2, 6, 32)
        output, mha_weights = model(x, need_weights=True)
        ref_output, ref_weights = reference(x, x, x, average_attn_weights=False)
    torch.testing.assert_close(output, ref_output, atol=1e-6, rtol=0)
    torch.testing.assert_close(mha_weights, ref_weights, atol=1e-6, rtol=0)
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    matrix = weights[0, 0].detach().numpy()
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    for row in range(6):
        for col in range(6):
            ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center",
                    color="white" if matrix[row, col] > .5 else "black")
    ax.set(xlabel="Key position", ylabel="Query position",
           title="Decoder-style causal attention\nFuture positions (above diagonal) = 0")
    fig.colorbar(im, ax=ax, label="Attention probability")
    fig.savefig(output_dir / "causal_attention.png", dpi=160)
    plt.close(fig)
    print("Causal weights, batch 0 / head 0:\n", weights[0, 0])
    return {"atol": 1e-6, "rtol": 0,
            "sdpa_max_abs_error": (actual - expected).abs().max().item(),
            "mha_max_abs_error": (output - ref_output).abs().max().item(),
            "mha_weights_max_abs_error": (mha_weights - ref_weights).abs().max().item(),
            "causal_future_mass": weights.triu(1).sum().item(),
            "family": "Decoder-style causal attention"}


def load_examples():
    examples = []
    with (ROOT / "data/raw/bayan_feedback.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["lang"] in {"ar", "en"} and row["lang"] not in {x["lang"] for x in examples}:
                examples.append({"feedback_id": row["feedback_id"], "lang": row["lang"],
                                 "text": normalize(mask_pii(row["text"]))})
            if len(examples) == 2:
                return examples
    raise ValueError("Expected at least one Arabic and one English Bayan example")


def pretrained_diagnostics(output_dir, cache_dir):
    options = dict(revision=REVISION, cache_dir=cache_dir)
    try:
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True, **options)
        model = AutoModel.from_pretrained(
            CHECKPOINT, local_files_only=True, attn_implementation="eager", **options
        )
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, **options)
        model = AutoModel.from_pretrained(CHECKPOINT, attn_implementation="eager", **options)
    model.eval()
    examples = load_examples()
    batch = tokenizer([x["text"] for x in examples], return_tensors="pt",
                      padding="max_length", truncation=True, max_length=64)
    valid = batch["attention_mask"].bool()
    assert (~valid).any(dim=1).all(), "Each example must contain padding for this diagnostic"
    with torch.inference_mode():
        masked = torch.stack(model(**batch, output_attentions=True).attentions)
        leaky_batch = {**batch, "attention_mask": torch.ones_like(batch["attention_mask"])}
        unmasked = torch.stack(model(**leaky_batch, output_attentions=True).attentions)
    # Shape is (layer, batch, head, query, key). Only real queries count as evidence.
    valid_queries = valid[None, :, None, :]
    pad_keys = (~valid)[None, :, None, None, :]

    def pad_mass(weights):
        return (weights * pad_keys).sum(-1)

    def mean_valid(values):
        return values.masked_select(valid_queries.expand_as(values)).mean().item()

    before, after = mean_valid(pad_mass(unmasked)), mean_valid(pad_mass(masked))
    assert before > 1e-6, "The deliberately unmasked run should expose pad leakage"
    assert after < 1e-7, "Regression: properly masked real queries must not attend to PAD"
    sep = batch["input_ids"].eq(tokenizer.sep_token_id)
    content = valid & ~sep & batch["input_ids"].ne(tokenizer.cls_token_id)
    length = valid.shape[1]
    positions = torch.arange(length)
    adjacent = (positions[:, None] - positions[None, :]).abs().eq(1)
    adjacent_keys = adjacent[None, None, None] & content[None, :, None, None, :]

    def per_head_mass(weights, keys):
        mass = (weights * keys).sum(-1)
        return (mass * content[None, :, None, :]).sum(dim=(1, 3)) / content.sum()

    adjacent_scores = per_head_mass(masked, adjacent_keys)
    sep_scores = per_head_mass(masked, sep[None, :, None, None, :])
    pad_scores = per_head_mass(unmasked, pad_keys)

    def strongest(scores):
        index = scores.argmax().item()
        layer, head = divmod(index, scores.shape[1])
        return {"layer": layer + 1, "head": head + 1, "mean_mass": scores[layer, head].item()}

    selected = {"adjacent_content": strongest(adjacent_scores),
                "sep_sink": strongest(sep_scores), "pad_leak": strongest(pad_scores)}
    for i, example in enumerate(examples):
        example["tokens"] = tokenizer.convert_ids_to_tokens(batch["input_ids"][i].tolist())
        example["real_tokens"] = valid[i].sum().item()
        example["pad_tokens"] = (~valid[i]).sum().item()
        example["pad_mass_unmasked"] = pad_mass(unmasked)[:, i, :, valid[i]].mean().item()
        example["pad_mass_masked"] = pad_mass(masked)[:, i, :, valid[i]].mean().item()

    # English token labels render directly; JSON retains both AR/EN token maps.
    sample = next(i for i, example in enumerate(examples) if example["lang"] == "en")
    tokens = examples[sample]["tokens"]
    real_length = examples[sample]["real_tokens"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), layout="constrained")
    panels = [("adjacent_content", masked, "Adjacent content: correct mask"),
              ("sep_sink", masked, "[SEP] sink: correct mask"),
              ("pad_leak", unmasked, "PAD leak: all keys allowed"),
              ("pad_leak", masked, "Same head: correct padding mask")]
    for ax, (kind, maps, title) in zip(axes.flat, panels):
        head = selected[kind]
        matrix = maps[head["layer"] - 1, sample, head["head"] - 1, :real_length]
        # Collapse PAD columns into their total mass, keeping real-token columns.
        compact = torch.cat([matrix[:, :real_length], matrix[:, real_length:].sum(-1, keepdim=True)], dim=-1)
        im = ax.imshow(compact.numpy(), aspect="auto", vmin=0, vmax=1, cmap="magma")
        ax.set_xticks(range(real_length + 1), tokens[:real_length] + ["ALL [PAD]"], rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(real_length), tokens[:real_length], fontsize=8)
        ax.set(xlabel="Key token (last column sums every PAD key)", ylabel="Real query token",
               title=f"{title}\nLayer {head['layer']}, head {head['head']} (1-based)")
        fig.colorbar(im, ax=ax, label="Attention mass", shrink=.8)
    fig.suptitle(f"Pretrained mBERT | {examples[sample]['feedback_id']} | eval mode, CPU\n"
                 "Heads ranked across the AR/EN pair; descriptive patterns, not semantic explanations", fontsize=13)
    fig.savefig(output_dir / "attention_maps.png", dpi=160)
    plt.close(fig)
    return {"checkpoint": CHECKPOINT, "revision": REVISION, "max_length": length,
            "mode": "pretrained weights, eval(), eager attention, CPU, float32",
            "pad_mass_definition": "sum over PAD keys, mean over real queries, heads and layers",
            "head_selection": "largest mean mass over content queries across the AR/EN pair; 1-based indices",
            "pad_mass_unmasked": before, "pad_mass_masked": after,
            "selected_heads": selected, "examples": examples,
            "limitations": "Two synthetic Bayan examples; descriptive diagnostics, not corpus-wide or causal explanations."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/lab2")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts/hf_cache")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    numerical = numerical_checks(args.output_dir)
    print("Numerical checks:", json.dumps(numerical, indent=2), flush=True)
    diagnostics = pretrained_diagnostics(args.output_dir, args.cache_dir)
    report = {"torch_version": torch.__version__, "transformers_version": transformers.__version__,
              "numerical": numerical, "diagnostics": diagnostics}
    (args.output_dir / "attention_diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"PAD mass: {diagnostics['pad_mass_unmasked']:.8f} -> {diagnostics['pad_mass_masked']:.8f}")
    print("Selected heads:", json.dumps(diagnostics["selected_heads"], indent=2))
    print(f"All Lab 2 anatomy checks passed. Evidence: {args.output_dir}")


if __name__ == "__main__":
    main()
