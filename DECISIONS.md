# Decision Records

## tokenizer
- Chosen checkpoint(s): `xlm-roberta-base` for the shared bilingual Lab 1–3 pipeline; retain `CAMeL-Lab/bert-base-arabic-camelbert-mix` only as the Arabic-centric candidate for Lab 4.
- Arabic fertility evidence: XLM-R = 1.672, versus mBERT = 2.153 and DistilBERT = 4.527. CAMeLBERT is lower at 1.405, but is not bilingual-efficient.
- English fertility evidence: XLM-R = 1.434, close to mBERT = 1.510 and DistilBERT = 1.298, while CAMeLBERT expands English heavily to 2.705.
- p95 length evidence: XLM-R = 21 AR / 23 EN tokens, balanced across both languages. CAMeLBERT = 20 / 38 and DistilBERT = 47 / 21, creating an avoidable language-specific serving budget.
- Operational trade-off / rationale: XLM-R had 0.00% AR UNK in this 7,200/4,800-row audit and the best balanced fertility/length profile. It is the lowest-friction shared checkpoint for bilingual topic/NER work; Lab 4 audits Arabic-centric candidates; the supplied validation split has no Gulf examples, so no Gulf-slice gain can be established.

## lab3-training
- Topic incumbent: word unigram/bigram TF-IDF + LinearSVC(C=1), trained only on the supplied training rows. Its validation macro-F1 is 1.0000. Do not weaken this baseline to manufacture the course's +8-point target.
- Transformer candidate: `xlm-roberta-base`, revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`, retaining the Lab 1 bilingual choice. Full fine-tuning, two epochs, AdamW at 2e-5, batch 16 with two accumulation steps, dynamic padding up to 64. Select the checkpoint on validation macro-F1, then evaluate the saved artifact on the frozen test once.
- Split decision: keep the provided 70/20/10 topic split and assert zero citizen overlap. However, 1,762 of 2,400 validation texts also occur in training after preprocessing. This limits conclusions about unseen formulations even with group separation.
- Final topic comparison: both models have test accuracy 1.0000 and fixed-eight-class macro-F1 0.5000; delta 0.00 points. The frozen test includes just four classes (300 rows each), so the other four contribute zero under the predeclared `zero_division=0` metric. Its mathematical ceiling is 0.5000. Retain TF-IDF as the measured quality/cost reference; XLM-R is a successfully trained comparison artifact, with no demonstrated gain on this test.
- NER: use the same base checkpoint with first-subword supervision and strict IOB2 entity-level seqeval metrics. Expand multiword CoNLL cells with BIO continuation labels. Split entire templates together after excluding unique reference values from grouping, because random sentence splitting would share almost identical templates across partitions.
- NER outcome: validation and held-out entity-F1 both 1.0000; the ≥0.80 course target is met on this small synthetic test. Keep the template/date/entity-coverage limitations with the result.
- QA: use pretrained `deepset/roberta-base-squad2`, revision `adc3b06f79f797d1c575d5479d6f5efe54a9e3b4`, because current QA contexts are English and the model supports SQuAD2 no-answer decisions. Calibrate the null-score difference threshold using only separate development contexts; preserve original text for character offsets.
- QA data issue: the original smoke file contains 12 answerable rows and zero nulls. Preserve it and report its result separately. A predeclared supplemental 9+3 set demonstrates null handling without relabelling the original test.
- Execution: local Apple MPS, float32. Transformers 4.43.4 requires eager attention for XLM-R. Save large model weights under ignored artifact folders and small measured reports under `artifacts/lab3/`.
- Detailed rationale and reproduction: [Lab 3 walkthrough](docs/LAB3_WALKTHROUGH.md).

## arabic-model
- Retain `xlm-roberta-base` as the shared bilingual incumbent. There is no held-out Gulf evidence to justify replacing it with a dialect-specific candidate.
- Candidates: fine-tuned `CAMeL-Lab/bert-base-arabic-camelbert-mix` and `CAMeL-Lab/bert-base-arabic-camelbert-da`; pinned revisions and identical candidate training settings are in `artifacts/lab4/arabic_bakeoff_protocol.json`.
- Coverage: all-Arabic validation equals MSA validation (1,200 rows); Gulf has zero validation rows. Unavailable Gulf scores are `null`, never zero performance or evidence of a win. Validation covers only 4 of 8 task labels, so fixed-label macro-F1 has a ceiling of 0.50.
- Measured candidate result: Mix and DA each reached MSA/all-Arabic validation accuracy 1.0000 and fixed-eight-class macro-F1 0.5000, matching XLM-R. No candidate demonstrated a gain on the available slice. Training/save/evaluation took 174.33 seconds for Mix and 173.85 seconds for DA on MPS.
- CI-backed verdict: unavailable for Gulf because there are no held-out Gulf examples. Aggregate synthetic-template performance cannot replace missing slice evidence; we do not claim a +4-point improvement.
- Normalization contract: `bayan_ar_v1` for the supplied golden rules; `camelbert_v1` is our conservative profile based on the CAMeLBERT preprocessing description. Same profile in candidate training and validation; retain separate PII-masked display text.
- Segmentation contract: CAMeL MLE `calima-msa-r13`, `d3tok`, no diacritics; preserve a map from each original word to its lexical stem. First-subword supervision is applied to that stem, other pieces are ignored. Version is recorded for segmented training/evaluation.
- Measured NER decision: keep the saved model's unsegmented path. On the paired 935-row validation set, LOCATION recall fell from 100% to 76.36% with D3; the +4-point target cannot be reached above a 100% baseline. A separate D3-consistent training run then recovered LOCATION recall and entity micro-F1 to 100% on the same 935 validation sentences. This is a 0-point gain against the original baseline, so retain the original model. The follow-up used validation for checkpoint selection and did not use the frozen test; it is development evidence, not independent test evidence.
- Details: [Lab 4 walkthrough](docs/LAB4_WALKTHROUGH.md) and [benchmarks](BENCHMARKS.md).

## search-min-score
- Threshold: **0.0065950584** on sigmoid cross-encoder scores, selected by balanced answerable/no-answer retention on supplied calibration queries.
- No-answer evidence: 20/20 empty-correct and 130/130 answerable retained, but only one unique negative text. No independent rejection-quality claim.
- Keep normalized cosine FAISS + multilingual reranking with pinned versions and checksums. Exact-ID retrieval targets remain unmet; do not relabel the sparse relevance sets to improve reported scores.
- The raw-vector ablation did not produce the expected collapse in this dataset. Retain the measured counterexample and investigate relevance coverage with the instructor.

## quantisation-split
- Topic artifact: dynamic INT8 ONNX; p99 4.84 ms, 38.70× versus padded fp32.
- NER artifact: dynamic INT8 ONNX; p99 4.86 ms, 32.10× versus padded fp32.
- Paired validation: all 2,400 classifier and 935 NER sentence predictions agree with saved fp32. Quality-tax intervals are zero on this synthetic validation sample. Frozen tests were not repeated.
- HTTP: bounded 8-item microbatches, 0.5 ms collection window; p99 33.36 ms at 16 concurrent clients over 60 seconds, zero errors. No output cache.
- Rollback: original PyTorch weights and ONNX fp32 exports remain locally available. Selected manifest and quality evidence are in `artifacts/lab7/`.

## architecture
- Bidirectional encoders support topic classification, token tagging, sentence embeddings and paired relevance scoring. Existing QA remains extractive; no generative decoder or optional extension was added.
- Retain the bilingual XLM-R task models. The Arabic DA candidate lacks Gulf evaluation coverage; it does not replace the bilingual classifier.
- Search and inference assets are loaded once; shared preprocessing and startup artifact checks prevent train/serve version skew. NER uses the source-word representation seen in training, with PII masking and preserved reference digits; offsets refer to returned masked text. Search runs on a dedicated CPU worker to avoid cross-thread OpenMP crashes.
- The sentiment baseline exists solely to exercise mandatory directional probes; its 0.3333 validation macro-F1 and all-tied directional predictions do not justify a production sentiment endpoint.
- Evidence: `BENCHMARKS.md`, `EVALUATION_REPORT.md`, and `docs/LABS_5_7_RUNBOOK.md`.
