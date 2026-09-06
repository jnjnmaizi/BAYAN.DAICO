# Decision Records

## tokenizer
- Chosen checkpoint(s): `xlm-roberta-base` for the shared bilingual Lab 1–3 pipeline; retain `CAMeL-Lab/bert-base-arabic-camelbert-mix` only as the Arabic-centric candidate for Lab 4.
- Arabic fertility evidence: XLM-R = 1.672, versus mBERT = 2.153 and DistilBERT = 4.527. CAMeLBERT is lower at 1.405, but is not bilingual-efficient.
- English fertility evidence: XLM-R = 1.434, close to mBERT = 1.510 and DistilBERT = 1.298, while CAMeLBERT expands English heavily to 2.705.
- p95 length evidence: XLM-R = 21 AR / 23 EN tokens, balanced across both languages. CAMeLBERT = 20 / 38 and DistilBERT = 47 / 21, creating an avoidable language-specific serving budget.
- Operational trade-off / rationale: XLM-R had 0.00% AR UNK in this 7,200/4,800-row audit and the best balanced fertility/length profile. It is the lowest-friction shared checkpoint for bilingual topic/NER work; Lab 4 will separately test whether CAMeLBERT's Arabic advantage produces a measured Gulf-slice gain.

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
