# Lab results and exact remaining items — 2026-09-10

Historical snapshot. For the subsequent accepted grouped review, see [review method](LAB6_REVIEW_METHOD.md). For newer retrieval experiments, see [Labs 3–5 follow-up](LABS_3_5_FOLLOWUP.md).

This status was checked against saved local evidence at code commit `dbf3323`. The current complete test suite passed **147 tests**, with two dependency deprecation warnings, in **7.75 seconds**. The earlier 140-test record in BENCHMARKS reflects the earlier Lab 7 run; seven review-related regression cases have since been added. No model training, frozen-test inference, requirements changes, source-data changes or human confirmations were performed for this status check.

## Lab 1 — Preprocessing and tokenizers

Implemented: versioned normalization, PII masking, sentence segmentation, six documented defect classes, five documented sentence spot checks, and four-tokenizer comparison on 12,000 feedback rows.

Results: 25/25 preprocessing golden tests; 60/60 PII recall. Selected XLM-R: Arabic/English fertility 1.672/1.434; p95 lengths 21/23. The complete comparison table is in [BENCHMARKS.md](../BENCHMARKS.md); explanations are in [NOTES.md](../NOTES.md) and [DECISIONS.md](../DECISIONS.md).

**Follow-up completed on 2026-09-10:** the previously missing [sequence-length histogram](../artifacts/lab1/token_length_histograms.png) and [exact frequency counts](../artifacts/lab1/tokenizer_audit.json) are now saved. The rerun reproduces the original table, all eight distributions have verified counts/p95 values, and the 26 original Lab 1 checks pass. [Run/interpretation guide](LAB1_HISTOGRAM_GUIDE.md).

## Lab 2 — Transformer attention

Implemented: scaled dot-product attention, multi-head attention, causal and padding masks, parameter audit and annotated attention maps.

Results: SDPA maximum absolute error 2.3841858e-7 versus PyTorch (required tolerance 1e-6); causal mass above diagonal zero; mean padding-attention mass 15.661530% without the mask and 0% with it. mBERT has 177,853,440 parameters; CAMeLBERT 109,081,344. Eleven original lab attention/audit tests passed in the recorded run.

**Status:** implemented and measured requirements met on the recorded examples.

Outputs: [parameter audit](../artifacts/lab2/parameter_audit.json), [attention diagnostics](../artifacts/lab2/attention_diagnostics.json), [attention maps](../artifacts/lab2/attention_maps.png), [causal map](../artifacts/lab2/causal_attention.png).

## Lab 3 — Topic classification, NER and QA

Implemented: TF-IDF baseline, grouped data splits with zero citizen overlap, XLM-R classifier training, BIO/subword alignment, NER training and constrained extractive QA.

Results:
- Topic baseline and transformer: validation macro-F1 1.0000; frozen-test macro-F1 0.5000 over the fixed eight-label task; frozen-test accuracy 1.0000 for both. Transformer gain is **0 points**, below the +8 target. Only four labels occur in the test, making 0.5000 the ceiling under this fixed-label metric. It does not mean half the predictions are wrong.
- NER: held-out entity-F1 1.0000, meeting the 0.80 target on 391 synthetic test sentences with two templates.
- QA: supplied smoke set 12/12 correct answerable spans; zero unanswerable cases in that file. Separate diagnostic: 9/9 spans and 3/3 nulls. The updated course README accepts 12 answerable cases; the older checklist's 9+3 mismatch remains documented.

**Remaining target:** +8 topic macro-F1 is not attainable above the current baseline ceiling on this test. Different valid evaluation evidence or a new evaluation protocol is required; the frozen test was not rerun to chase the target.

Outputs: [baseline](../artifacts/lab3/tfidf_baseline.json), [classifier](../artifacts/lab3/topic_classifier.json), [NER](../artifacts/lab3/ner.json), [QA](../artifacts/lab3/qa.json), [target audit](../artifacts/lab3/topic_target_audit.json), [QA updated requirement assessment](../artifacts/lab3/qa_course_update.json).

## Lab 4 — Arabic preprocessing and segmentation

Implemented: two Arabic normalization profiles, safe display text, dialect audit, real CAMeL D3 segmentation with source-word alignment, and a separately trained D3 NER model.

Results: 30/30 Arabic golden cases. Original LOCATION recall 100%; feeding D3 input to the original model reduced it to 76.36%; training consistently with D3 recovered it to 100%. Improvement over the original baseline is **0 points**, not +4.

The earlier optional Mix/DA bake-off was also measured: both have MSA accuracy 1.0 and fixed-eight-label macro-F1 0.5, tying the bilingual model. All 4,800 Gulf rows are training rows; no Gulf validation/test comparison is available.

**Remaining required target:** +4 LOCATION recall is not attainable above the original 100% baseline on this validation set. **Optional/deferred:** further Arabic model bake-offs, MARBERT and Gulf-gain experiments without independent Gulf data.

Outputs: [Lab 4 summary](../artifacts/lab4/summary.json), [dialect distribution](../artifacts/lab4/dialect_audit.json), [D3 training comparison](../artifacts/lab4/ner_d3_comparison.json), [Arabic model comparison](../artifacts/lab4/arabic_bakeoff.json).

## Lab 5 — Bilingual search

Implemented: persisted normalized FAISS index covering 20,000 cases, metadata/checksum/version manifest, multilingual bi-encoder and cross-encoder, top-10 metrics, cross-language analysis, threshold calibration and the unnormalized-vector ablation.

Results: normalized stage-1 recall@10/MRR@10 0.002564/0.002564; reranked recall@10 **0.023077 (2.31%)**, MRR@10 **0.026484**, versus targets 0.80/0.70. Reranking adds 0.023919 MRR. Cross-language relevant-ID recall is zero in the top ten. Same-minus-cross MRR gap is 0.026484.

The threshold rejects 20/20 negatives and retains 130/130 answerable queries, but the negatives contain one unique text and were used for calibration. This is not independent rejection evidence.

**Remaining targets:** recall/MRR are unmet. Repeated texts and sparse exact-ID labels complicate evaluation but do not establish that the implementation is optimal. Further retrieval and relevance-judgement investigation is needed. The expected unnormalized-vector metric collapse was **not observed**; raw-vector scores were higher than normalized stage 1 in this run. No labels were changed to force the expected result.

Outputs: [retrieval metrics](../artifacts/lab5/retrieval.json), [per-query rankings](../artifacts/lab5/query_results.json), [data audit](../artifacts/lab5/data_audit.json), [index manifest](../artifacts/lab5/index_manifest.json).

## Lab 6 — Evaluation and error review

Implemented: bootstrap and paired intervals, 16 topic slices, behavioural tests, report generator, three model cards, per-entry assistant review and a resumable human-review tool following the course workflow.

Results: topic invariance 200/200; minimum-functionality tests 15/16 (93.75%). Arabic DA MFT is 8/8 on Arabic but 9/16 across both languages. A separate sentiment baseline passes 200/200 directional checks through 200 unchanged predictions; its validation macro-F1 is only 0.3333, so this does not establish negation understanding.

Assistant review: all 120 entries annotated; 45 exact texts; all supplied errors are parks predicted as roads. Scenarios: playground maintenance 42, park accessibility 46, irrigation 32. Assistant histogram and three proposed follow-ups are saved. Accounting ceilings from replacing wrong predictions with gold labels are explicitly not model improvements.

**Historical snapshot:** this file predates the completed 120/120 individual review. The current review state is recorded in `artifacts/lab6/human_error_review.json` and `EVALUATION_REPORT.md`.

Outputs: [evaluation report](../EVALUATION_REPORT.md), [Lab 6 summary](../artifacts/lab6/summary.json), [short assistant review](LAB6_QUICK_REVIEW.md), [all assistant annotations](../artifacts/lab6/assistant_error_review.json), [assistant histogram](../artifacts/lab6/assistant_error_taxonomy.png), [human review state](../artifacts/lab6/human_error_review.json), [GitHub workflow](LAB6_GITHUB_WORKFLOW.md), [model cards](../model_cards).

## Lab 7 — CPU optimization and API

Implemented: original baseline timings, dynamic-padding comparison, ONNX and INT8 export for classifier/NER, paired validation checks, model selection, fp32 rollback, startup canaries, bounded microbatching and all service endpoints.

Results: classifier padded PyTorch p99 187.39 ms; dynamic PyTorch 35.97 ms; ONNX 9.23 ms; selected INT8 **4.84 ms**, a **38.70×** speed-up over the padded p99. NER INT8 p99 **4.86 ms**. All 2,400 saved classifier and 935 NER validation sentence predictions match; measured paired quality tax and its interval are zero on these data.

Final HTTP: **33.36 ms p99**, 16 concurrent clients, 60 seconds, 33,259 requests, zero errors, startup canaries passed. Original unbatched and 16-item microbatch attempts are retained. Selected batching is eight items with a 0.5 ms window and no prediction cache. All five endpoints, including health and Arabic/English combined analysis, were checked.

**Status:** the measured Lab 7 targets are met. The HTTP result uses the supplied fixed request text; it does not establish the same latency for all payloads or concurrent search workloads. Quantization quality is validated on the synthetic validation data, not a new frozen test.

Outputs: [selected models](../artifacts/lab7/selection.json), [HTTP result](../artifacts/lab7/http_load.json), [API responses](../artifacts/lab7/api_integration.json), [classifier quality](../artifacts/lab7/classifier_int8_quality.json), [NER quality](../artifacts/lab7/ner_int8_quality.json), and the full ladder in [BENCHMARKS.md](../BENCHMARKS.md).

## Where the large runnable artifacts are

These remain on this Mac and are excluded from the small results archive and GitHub:

- `artifacts/topic_classifier/model.safetensors`
- `artifacts/ner/model.safetensors`
- `artifacts/ner_d3/model.safetensors`
- `artifacts/search/cases.faiss` plus metadata and raw vectors
- `artifacts/onnx/classifier/fp32.onnx`, `int8.onnx`
- `artifacts/onnx/ner/fp32.onnx`, `int8.onnx`

Optional work deferred: further Arabic/MARBERT comparisons, extra model or hyperparameter sweeps, and new capstone extensions. These are separate from the unfinished mandatory items listed above.
