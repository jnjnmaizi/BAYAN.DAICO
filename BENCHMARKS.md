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
| TF-IDF + LinearSVC | macro-F1 over all 8 task classes | 1.0000 | 0.5000 | 0.23 s, CPU |
| XLM-R topic classifier | macro-F1 over all 8 task classes | 1.0000 | 0.5000 | 470.31 s, MPS |
| XLM-R NER | strict IOB2 entity-F1 | 1.0000 | 1.0000 | 245.71 s, MPS |
| Pretrained RoBERTa SQuAD2 QA | exact character span / null | development: 144/144 | supplied smoke: 12/12 answerable, 0 null cases | no new training |

Measured on 2026-09-08; seed 42, Python 3.12.7, PyTorch 2.14.0, Transformers 4.43.4. Topic training uses the same Lab 1 preprocessing for both models, with 8,400 train / 2,400 validation / 1,200 test rows and **zero citizen overlap**. Test evaluation happened once per saved topic model after selecting the transformer checkpoint on validation.

**Topic metric coverage:** the frozen test contains only four of the eight classes: digital_services, lighting, parks and water, 300 rows each. Both models have **test accuracy 1.0000**. The predeclared macro-F1 averages all eight training labels with `zero_division=0`, so absent classes contribute zero and the maximum achievable score on this test is **0.5000**. Measured transformer delta: **0.00 percentage points**; the course's +8-point target is not met. Do not interpret 0.5000 here as 50% incorrect predictions or infer performance on the absent classes.

- Topic data limitation: 1,762/2,400 validation texts also occur in training after preprocessing. The synthetic templates make high scores easy; this is not evidence of generalisation to new complaint formulations.
- NER policy: group by full sentence template with reference values excluded from the grouping key. Train 2,674 rows/12 templates, validation 935/4, test 391/2; zero template overlap. Multiword source cells are expanded with BIO continuation tags; special and non-first subword pieces are ignored in loss/evaluation.
- NER result: test precision/recall/F1 all 1.0000, meeting the ≥0.80 target on the 391-row held-out set. The test contains only two templates; there are no ORGANISATION examples and no meaningful unseen-date coverage.
- NER export recovery: the initial report export failed on a NumPy support count after saving weights and frozen-test aggregates. JSON conversion was fixed; metadata was recovered from the original training log, checkpoint and saved validation predictions. The frozen aggregate was reused unchanged, with no retraining or repeat test inference. Per-entity test details were not retained by that initial run.
- QA calibration: 144 unique questions from 36 development contexts, all disjoint from supplied and supplemental evaluation contexts. Chosen null threshold **0.0**; total inference/calibration/evaluation time **6.39 s, CPU**, excluding model loading.
- QA supplied-data discrepancy: the official smoke file has **12 answerable rows (3 unique questions), zero unanswerable rows**, so its documented 9+3 requirement cannot be verified from that file.
- Separate predeclared QA diagnostic: **9/9 exact answer spans + 3/3 correct nulls** from three other contexts; this is supplemental evidence, not a replacement of the supplied smoke set. Source data and expected answers remain unchanged.
- Lab 1–3 unit/regression checks: **63 passed**, including all 11 original Lab 3 contracts and 15 additional Lab 3 regressions.
- Evidence: [baseline](artifacts/lab3/tfidf_baseline.json), [classifier](artifacts/lab3/topic_classifier.json), [NER](artifacts/lab3/ner.json), [QA](artifacts/lab3/qa.json), [QA protocol](artifacts/lab3/qa_protocol.json), [data audit](artifacts/lab3/data_audit.json).
- Rationale and commands: [Lab 3 walkthrough](docs/LAB3_WALKTHROUGH.md). Large trained models remain under ignored `artifacts/topic_classifier/` and `artifacts/ner/`.

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
