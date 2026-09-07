# BENCHMARKS

> Fill these tables from **your own runs**. Do not copy course reference numbers.

## Lab 1 — Tokenizer audit
| Tokenizer | AR fertility | EN fertility | AR p95 len | EN p95 len | AR UNK rate |
|---|---:|---:|---:|---:|---:|
| mBERT | 2.153 | 1.510 | 27.0 | 25.0 | 0.45% |
| XLM-R | 1.672 | 1.434 | 21.0 | 23.0 | 0.00% |
| CAMeLBERT | 1.405 | 2.705 | 20.0 | 38.0 | 0.80% |
| DistilBERT | 4.527 | 1.298 | 47.0 | 21.0 | 0.22% |

- Audit corpus: 7,200 Arabic and 4,800 English feedback rows; content pieces were used for fertility and special tokens were included in lengths.
- Golden preprocessing: 25 / 25 passed
- PII masking recall: 60 / 60 = 100%

## Lab 2 — Attention and mask diagnostics

Measured on 2026-09-07 with Python 3.12.7, PyTorch 2.14.0 and Transformers 4.43.4, CPU float32. Numerical experiment seed: 42; comparison tolerance: `atol=1e-6, rtol=0`.

| Numerical check | Measured result |
|---|---:|
| SDPA maximum absolute error versus PyTorch | 2.3841858e-7 |
| MHA maximum absolute output error versus PyTorch | 8.9406967e-8 |
| MHA maximum absolute weight error versus PyTorch | 5.9604645e-8 |
| Causal attention mass above the diagonal | 0 |
| Attention and parameter-audit unit tests | 11 passed |

Pretrained mBERT (`3f076fdb1ab68d5b2880cb87a0886f315b8146f8`), `eval()`, eager attention, max length 64. PAD mass = sum over PAD keys, averaged over **real query positions**, 12 heads and 12 layers. The combined mean weights queries equally, rather than averaging the two per-example percentages.

| Example | Real tokens | PAD tokens | PAD mass without mask | PAD mass with mask |
|---|---:|---:|---:|---:|
| FB-000001 / AR | 41 | 23 | 12.670256% | 0.000000% |
| FB-000004 / EN | 21 | 43 | 21.501637% | 0.000000% |
| Both / real-query-weighted mean | 62 | 66 | 15.661530% | 0.000000% |

- The deliberately broken run allows every key; the corrected run passes the tokenizer's attention mask. The script asserts unmasked PAD mass >1e-6 and corrected mass <1e-7.
- Only two synthetic Bayan examples were inspected; no quality or corpus-wide claim is implied.
- Exact metrics and input/token provenance: [attention_diagnostics.json](artifacts/lab2/attention_diagnostics.json). Annotated [attention maps](artifacts/lab2/attention_maps.png) and [causal map](artifacts/lab2/causal_attention.png).
- Parameter accounting and head interpretation: [NOTES.md](NOTES.md#lab-2--parameter-audit).
- Reproduce with `source .venv/bin/activate` then `make lab2`.

## Lab 3 — Models
| Model | Metric | Validation | Frozen test | Train time |
|---|---|---:|---:|---:|
| TF-IDF + LinearSVC | macro-F1 | | | |
| Topic classifier | macro-F1 | | | |
| NER | entity-F1 | | | |
| QA | span/null smoke | | | |

## Lab 4 — Arabic model bake-off
| Checkpoint | macro-F1 all | Gulf | MSA | AR fertility |
|---|---:|---:|---:|---:|
| multilingual incumbent | | | | |
| Arabic dialect-aware | | | | |
| optional third model | | | | |

## Lab 5 — Search
| Configuration | recall@10 | MRR@10 | p50 latency/query |
|---|---:|---:|---:|
| bi-encoder only | | | |
| + cross-encoder rerank | | | |
| cross-lingual slice | | | |

- no-answer empty-correct: ___ / 20
- cross-lingual gap: ___

## Lab 6 — Evaluation
| Model | Aggregate macro-F1 [CI] | Gulf [CI] | Invariance pass | MFT pass |
|---|---|---|---:|---:|
| topic classifier | | | | |
| dialect-aware | | | | |

- paired comparison verdict:
- error taxonomy top categories:
- top-3 prioritised fixes:

## Lab 7 — Optimisation ladder
| Rung | p50 | p99 | quality metric / paired Δ | Artefact size |
|---|---:|---:|---|---:|
| fp32 torch @512 padded | | | | |
| fp32 torch @128 dynamic | | | | |
| ONNX fp32 @128 | | | | |
| ONNX INT8 @128 | | | | |

- HTTP p99, 16 concurrent:
- classifier quantisation decision:
- NER quantisation decision:
