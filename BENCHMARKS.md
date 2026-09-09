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

**Target feasibility audit (2026-09-09):** the saved baseline already reaches the 0.5000 ceiling, so the maximum possible improvement is **0.00 points**. This was computed without new training or test inference, with data/report hashes checked. See [machine-readable audit](artifacts/lab3/topic_target_audit.json), `python scripts/topic_target_audit.py`, and [resolution steps / instructor draft](docs/LAB3_STATUS.md).

- Topic data limitation: 1,762/2,400 validation texts also occur in training after preprocessing. The synthetic templates make high scores easy; this is not evidence of generalisation to new complaint formulations.
- NER policy: group by full sentence template with reference values excluded from the grouping key. Train 2,674 rows/12 templates, validation 935/4, test 391/2; zero template overlap. Multiword source cells are expanded with BIO continuation tags; special and non-first subword pieces are ignored in loss/evaluation.
- NER result: test precision/recall/F1 all 1.0000, meeting the ≥0.80 target on the 391-row held-out set. The test contains only two templates; there are no ORGANISATION examples and no meaningful unseen-date coverage.
- NER export recovery: the initial report export failed on a NumPy support count after saving weights and frozen-test aggregates. JSON conversion was fixed; metadata was recovered from the original training log, checkpoint and saved validation predictions. The frozen aggregate was reused unchanged, with no retraining or repeat test inference. Per-entity test details were not retained by that initial run.
- QA calibration: 144 unique questions from 36 development contexts, all disjoint from supplied and supplemental evaluation contexts. Chosen null threshold **0.0**; total inference/calibration/evaluation time **6.39 s, CPU**, excluding model loading.
- Historical QA supplied-data discrepancy (before the 2026-09-09 README sync): the official smoke file has **12 answerable rows (3 unique questions), zero unanswerable rows**, so its documented 9+3 requirement cannot be verified from that file.
- Updated course target (README `7949de0`, synced 2026-09-09): **12/12 answerable spans with zero null cases — met** by the saved result above. This is a reassessment of existing evidence, with no new inference. Historical JSON reports retain their original 9+3 assessment; see [updated assessment](artifacts/lab3/qa_course_update.json) and [source reconciliation](docs/COURSE_SYNC.md). Zero null cases do not establish no-answer performance.
- Separate predeclared QA diagnostic: **9/9 exact answer spans + 3/3 correct nulls** from three other contexts; this is supplemental evidence, not a replacement of the supplied smoke set. Source data and expected answers remain unchanged.
- Lab 1–3 unit/regression checks: **63 passed**, including all 11 original Lab 3 contracts and 15 additional Lab 3 regressions.
- Evidence: [baseline](artifacts/lab3/tfidf_baseline.json), [classifier](artifacts/lab3/topic_classifier.json), [NER](artifacts/lab3/ner.json), [QA](artifacts/lab3/qa.json), [QA protocol](artifacts/lab3/qa_protocol.json), [data audit](artifacts/lab3/data_audit.json).
- Rationale and commands: [Lab 3 walkthrough](docs/LAB3_WALKTHROUGH.md). Large trained models remain under ignored `artifacts/topic_classifier/` and `artifacts/ner/`.

## Lab 4 — Arabic profiles, segmentation and model comparison

Measured locally on 2026-09-09 with Python 3.12.7, PyTorch 2.14.0, Transformers 4.43.4, CAMeL Tools 1.5.7. Candidate fine-tuning used Apple MPS; the fixed-model NER ablation used CPU.

| Model | All Arabic validation macro-F1 | Gulf | MSA macro-F1 | Validation accuracy | Train/save/eval seconds |
|---|---:|---|---:|---:|---:|
| Existing XLM-R | 0.5000 | N/A: 0 rows | 0.5000 | 1.0000 | reused saved validation predictions |
| CAMeLBERT-Mix | 0.5000 | N/A: 0 rows | 0.5000 | 1.0000 | 174.33 |
| CAMeLBERT-DA | 0.5000 | N/A: 0 rows | 0.5000 | 1.0000 | 173.85 |

All-Arabic validation consists of 1,200 MSA rows and zero Gulf rows; only four of eight task labels are represented, so the fixed-label macro-F1 ceiling is 0.5000. Both new models and the saved incumbent reach that ceiling. There is **no measurable Gulf delta** and no evidence to replace XLM-R; the +4-point Gulf target is unassessable, not achieved. Candidate runs used 6,000 Arabic training rows, the same pinned Mix/DA revisions, two requested epochs, seed 42, LR 2e-5, batch 16 / accumulation 2, max length 64 and the conservative `camelbert_v1` profile. The optimizer completed 374 steps (1.9947 epochs as reported by Trainer). This is model-selection evidence on validation, not a frozen-test result or an isolated causal estimate of dialectal pretraining.

| NER paired validation metric (935 original sentences) | Original input | CAMeL D3 input |
|---|---:|---:|
| LOCATION precision | 1.0000 | 1.0000 |
| LOCATION recall | 1.0000 | 0.763636 |
| LOCATION F1 | 1.0000 | 0.865979 |
| Overall strict entity micro-F1 | 1.0000 | 0.940909 |

- LOCATION recall delta: **−23.6364 points**; +4 target not met. Baseline recall is already 100%, so maximum positive headroom is zero. Keep the saved NER model's unsegmented path. Runtime: 6.55 seconds, CPU, excluding model loading.
- Evaluation used the same original word/BIO boundaries, projecting each D3 lexical stem's first subword prediction back to its source word. The saved original validation predictions were reused. No frozen-test inference or NER retraining was performed.
- Normalization: **30/30** original golden rows passed (10 unique cases). **14** new Lab 4 regressions cover the second profile, display/PII preservation, morphology alignment, empty Gulf slices and strided BERT weight serialization. Labs 1–4 total: **118 passed**.
- Integration check: all **4,000** source NER sentences retained exactly their original supervised labels through D3 + subword alignment, with no truncation.
- Dialect metadata: Gulf 4,800/7,200 Arabic rows (66.67%), MSA 2,400 (33.33%); every Gulf row is in training and the frozen topic test has no Arabic rows.
- Evidence: [dialect audit](artifacts/lab4/dialect_audit.json), [profile and segmentation examples](artifacts/lab4/preprocessing_examples.json), [NER ablation](artifacts/lab4/ner_segmentation.json), [Mix/DA comparison](artifacts/lab4/arabic_bakeoff.json), [summary](artifacts/lab4/summary.json).
- Reproduction and explanations: [Lab 4 walkthrough](docs/LAB4_WALKTHROUGH.md). Large models stay under ignored `artifacts/lab4_models/`; the published JSON files contain measured evidence only.

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
