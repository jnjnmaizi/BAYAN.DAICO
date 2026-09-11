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
- Histogram follow-up (2026-09-10): saved [four-panel length distributions](artifacts/lab1/token_length_histograms.png) and [exact counts/statistics](artifacts/lab1/tokenizer_audit.json). All eight language/tokenizer distributions match the original table; full untruncated lengths include special tokens, and language histograms are independently normalized to 100%. Source data unchanged. Original Lab 1 checks: **26 passed**. [Run and interpretation guide](docs/LAB1_HISTOGRAM_GUIDE.md).

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

**Target feasibility audit (2026-09-09):** the saved baseline already reaches the 0.5000 ceiling, so the maximum possible improvement is **0.00 points**. This was computed without new training or test inference, with data/report hashes checked. See [machine-readable audit](artifacts/lab3/topic_target_audit.json), `python scripts/topic_target_audit.py`, and [resolution steps](docs/LAB3_STATUS.md).

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
- Evaluation used the same original word/BIO boundaries, projecting each D3 lexical stem's first subword prediction back to its source word. The saved original validation predictions were reused. This initial ablation performed no NER retraining or frozen-test inference.
- D3-consistent training follow-up: three epochs on 2,674 training sentences, **282.92 seconds on MPS**. On the same 935 validation sentences, LOCATION precision/recall/F1 and overall entity micro-F1 all reached **1.0000**. Gain against the original model is **0 points**; +4 target remains unmet. The best checkpoint was selected on validation; no frozen-test inference was performed. Saved weights reloaded on CPU with finite logits and matching predictions on one validation example from each of four templates. See [training](artifacts/lab4/ner_d3_training.json), [comparison](artifacts/lab4/ner_d3_comparison.json), and [reload check](artifacts/lab4/ner_d3_reload_check.json).
- Normalization: **30/30** original golden rows passed (10 unique cases). **14** new Lab 4 regressions cover the second profile, display/PII preservation, morphology alignment, empty Gulf slices and strided BERT weight serialization. The initial Labs 1–4 run had **118 passed**. The D3 follow-up added two comparison regressions (16 Lab 4 regressions now); the affected Lab 4, NER alignment and Lab 3 regression suites passed **43 tests**.
- Integration check: all **4,000** source NER sentences retained exactly their original supervised labels through D3 + subword alignment, with no truncation.
- Dialect metadata: Gulf 4,800/7,200 Arabic rows (66.67%), MSA 2,400 (33.33%); every Gulf row is in training and the frozen topic test has no Arabic rows.
- Evidence: [dialect audit](artifacts/lab4/dialect_audit.json), [profile and segmentation examples](artifacts/lab4/preprocessing_examples.json), [NER ablation](artifacts/lab4/ner_segmentation.json), [Mix/DA comparison](artifacts/lab4/arabic_bakeoff.json), [summary](artifacts/lab4/summary.json).
- Reproduction and explanations: [Lab 4 walkthrough](docs/LAB4_WALKTHROUGH.md). Large models stay under ignored `artifacts/lab4_models/` and `artifacts/ner_d3/`; the published JSON files contain measured evidence only.

## Lab 5 — Search

**Completion decision:** Labs 3–5 have additional evidence documented. The official numeric results in this section are preserved and are not rewritten as target scores.

| Configuration | recall@10 | MRR@10 | p50 stage latency |
|---|---:|---:|---:|
| normalized bi-encoder | 0.002564 | 0.002564 | 7.95 ms |
| + multilingual cross-encoder | 0.023077 | 0.026484 | +49.49 ms |
| unnormalized-vector ablation | 0.015385 | 0.008034 | not separately benchmarked |

- Full 20,000-case corpus; 130 answerable labelled queries, 50 candidates, top 10. MPS inference; encoder and reranker revisions pinned in the manifest. Original exact-ID labels remain unchanged.
- Reranking MRR lift: **+0.023919**. The official recall/MRR values remain below the numeric targets; the additional evidence is reported separately. Corpus contains 5,401 unique texts spread across 20,000 distinct case IDs; labels name only three IDs per query.
- Within-query language comparison: same-language relevant-ID recall@10 **0.046154**, cross-language **0.000000**; same-minus-cross MRR gap **0.026484**. All queries have mixed-language relevance sets.
- Empty-correct: **20/20**, answerable retained **130/130**, threshold **0.0065950584**. This is calibration on the supplied set, whose 20 no-answer rows contain only one unique text; no independent threshold test is available.
- The expected unnormalized-vector metric collapse was **not observed**. Raw vectors scored higher than normalized stage 1 under sparse exact-ID labels. Production still enforces normalized vectors and validates checksums.
- Evidence: [retrieval](artifacts/lab5/retrieval.json), [data audit](artifacts/lab5/data_audit.json), [manifest](artifacts/lab5/index_manifest.json).
- Additional evidence and record: [alternative evidence](docs/ALTERNATIVE_EVIDENCE.md), [JSON record](artifacts/alternative_evidence.json).
- Dataset/relevance audit: all 130 answerable queries use relevant IDs exactly 8 corpus rows apart, matching the repeating 8-topic cycle. Under the supplementary verbatim-duplicate definition, the saved reranker reaches recall@10 **0.951456** and MRR@10 **0.683545** on 103 queries. See [Lab 5 dataset audit](docs/LAB5_DATASET_AUDIT.md).

## Lab 6 — Evaluation

| Model | Aggregate macro-F1 [95% CI] | Gulf | Invariance | MFT |
|---|---|---|---:|---:|
| topic classifier | 1.0000 [1.0000, 1.0000] | unavailable | 200/200 | 15/16 |
| Arabic DA candidate | 0.5000 [0.5000, 0.5000] | unavailable | 200/200 | 9/16 bilingual; 8/8 Arabic |

- Topic report: **16 slices**; DA report: 10 available/empty slices. Fixed eight-label macro-F1 and 500 seeded bootstrap draws resampling citizen groups. DA validation contains only four task classes, so its ceiling is 0.5.
- Paired topic-minus-DA difference on the same 1,200 Arabic rows: **0.0000 [0.0000, 0.0000]**. No Gulf comparison can be established.
- Separate sentiment baseline: **200/200** directional checks, all **200 ties**; validation macro-F1 **0.3333**. This does not demonstrate negation sensitivity and is not a topic-model score.
- Error analysis: **120 sampled entries**, comprising 45 distinct text/label/prediction groups. Scenarios: playground maintenance **42**, park accessibility **46**, irrigation **32**. All observed errors are parks → roads.
- Human review: **120/120 individual entries confirmed**, with 45 grouped decisions and reviewer notes retained as a compact summary. [Completed worksheet](docs/LAB6_SHORT_REVIEW.md), [individual record](artifacts/lab6/human_error_review.json), and [review method](docs/LAB6_REVIEW_METHOD.md).
- Three model cards created. Full results and limitations: [evaluation report](EVALUATION_REPORT.md), [review worksheet](docs/LAB6_ERROR_REVIEW.md), [model cards](model_cards).

## Lab 7 — Optimisation ladder

CPU measurements on this Mac, four threads, batch 1, 10 warm-ups and the same 200 seeded samples from the supplied 2,000-row production mix. Bare timings exclude tokenization. The fp32 baseline was measured before ONNX export.

| Task / rung | p50 ms | p99 ms | Validation quality tax [95% CI] | Artifact MiB |
|---|---:|---:|---|---:|
| classifier / torch padded_512 | 146.21 | 187.39 | baseline | 1060.7 |
| classifier / torch dynamic_128 | 30.10 | 35.97 | baseline | 1060.7 |
| classifier / onnx dynamic_128 | 7.51 | 9.23 | 0.0000 [0.0000, 0.0000] | 1060.9 |
| classifier / int8 dynamic_128 | 3.26 | 4.84 | 0.0000 [0.0000, 0.0000] | 816.8 |
| ner / torch padded_512 | 143.68 | 155.85 | baseline | 1058.5 |
| ner / torch dynamic_128 | 29.12 | 31.18 | baseline | 1058.5 |
| ner / onnx dynamic_128 | 7.08 | 8.49 | 0.0000 [0.0000, 0.0000] | 1058.7 |
| ner / int8 dynamic_128 | 3.28 | 4.86 | 0.0000 [0.0000, 0.0000] | 816.2 |

- Classifier decision: **INT8**, bare p99 **4.84 ms**, **38.70×** faster than padded fp32 p99; quality tax 0 points, upper paired-bootstrap bound 0. All 2,400 saved validation predictions match.
- NER decision: **INT8**, bare p99 **4.86 ms**, **32.10×** faster than padded fp32 p99. All 935 saved validation sentence predictions match. The quality interval resamples only four NER template groups, so external generalisation remains unproven.
- Original PyTorch and ONNX fp32 weights retained for rollback. Dynamic quantization covers constant MatMul weights; the large embedding tables remain fp32.

| HTTP configuration, 16 concurrent clients / 60 seconds | p50 ms | p99 ms | Requests | Errors |
|---|---:|---:|---:|---:|
| unbatched | 21.86 | 325.78 | 22110 | 0 |
| 16-item / 1 ms microbatch | 29.98 | 42.77 | 31723 | 0 |
| 8-item / 0.5 ms microbatch (selected) | 28.74 | 33.36 | 33259 | 0 |

- Selected HTTP p99 **33.36 ms ≤40 ms**, zero request errors, startup canaries green. No prediction caching. The load driver uses persistent httpx clients with the same concurrency, duration and fixed payload as the supplied hey command.
- Classifier bare p99, speed-up, quality-tax and HTTP targets all met on this measured machine/run. HTTP performance for longer or more varied payloads is not established by this fixed-payload test.
- Startup checks cover selected model checksums, tokenizer/config hashes, preprocessing version, PII handling and Arabic/English label probes.
- Final unit/contract suite: **140 passed**. Evidence and runnable commands: [runbook](docs/LABS_5_7_RUNBOOK.md), [selection](artifacts/lab7/selection.json), [HTTP result](artifacts/lab7/http_load.json).
