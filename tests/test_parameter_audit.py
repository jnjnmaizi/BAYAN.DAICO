"""Check disjoint audit buckets against an independent tiny BERT calculation."""
import importlib.util
from pathlib import Path

import torch
from transformers import BertConfig, BertModel


def test_bert_parameter_buckets():
    spec = importlib.util.spec_from_file_location(
        "parameter_audit", Path(__file__).resolve().parents[1] / "scripts/parameter_audit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = BertConfig(vocab_size=31, hidden_size=12, intermediate_size=19,
                        num_hidden_layers=2, num_attention_heads=3,
                        max_position_embeddings=17, type_vocab_size=2)
    with torch.device("meta"):
        model = BertModel(config)
    result = module.count_parameters(model)
    expected = {
        "embeddings": (31 + 17 + 2) * 12,
        "attention": 2 * 4 * (12 * 12 + 12),
        "ffn": 2 * (12 * 19 + 19 + 19 * 12 + 12),
        "norms": (1 + 2 * 2) * 2 * 12,
        "pooler": 12 * 12 + 12,
        "other": 0,
    }
    assert result["buckets"] == expected
    assert result["total_params"] == sum(expected.values())
    assert result["embeddings_pct"] == 100 * expected["embeddings"] / sum(expected.values())
