# Decision Records

## tokenizer
- Chosen checkpoint(s): `xlm-roberta-base` for the shared bilingual Lab 1–3 pipeline; retain `CAMeL-Lab/bert-base-arabic-camelbert-mix` only as the Arabic-centric candidate for Lab 4.
- Arabic fertility evidence: XLM-R = 1.672, versus mBERT = 2.153 and DistilBERT = 4.527. CAMeLBERT is lower at 1.405, but is not bilingual-efficient.
- English fertility evidence: XLM-R = 1.434, close to mBERT = 1.510 and DistilBERT = 1.298, while CAMeLBERT expands English heavily to 2.705.
- p95 length evidence: XLM-R = 21 AR / 23 EN tokens, balanced across both languages. CAMeLBERT = 20 / 38 and DistilBERT = 47 / 21, creating an avoidable language-specific serving budget.
- Operational trade-off / rationale: XLM-R had 0.00% AR UNK in this 7,200/4,800-row audit and the best balanced fertility/length profile. It is the lowest-friction shared checkpoint for bilingual topic/NER work; Lab 4 will separately test whether CAMeLBERT's Arabic advantage produces a measured Gulf-slice gain.

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
- Incumbent:
- Candidate:
- All/Gulf/MSA evidence:
- CI-backed verdict:
- Segmentation contract:

## search-min-score
- Threshold:
- No-answer evidence:
- False-positive / false-negative trade-off:

## quantisation-split
- Topic artefact:
- NER artefact:
- Latency evidence:
- Paired quality-tax evidence:
- Rollback artefact retained:

## architecture
- Encoder/decoder rationale by task:
- Multilingual vs Arabic-centric rationale:
- Evidence used:
