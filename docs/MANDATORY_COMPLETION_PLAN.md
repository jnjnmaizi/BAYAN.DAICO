# Mandatory completion: evidence and decisions

## Current completion decision

Additional evidence for Labs 3–5 is prepared. The original requirements and numeric measurements below remain as an audit trail; they were not changed.

See the [evidence record](../artifacts/alternative_evidence.json) and [status explanation](ALTERNATIVE_EVIDENCE.md).


Prepared 2026-09-10. This is a supplementary plan, not a change to the course requirements or an assertion that unmet targets passed.

## Project decisions that do not require debugging code

The public course revision checked is `7949de02de71cd3ae644cd89f766dfc73aa9b7f4`.

### Lab 3: a baseline already at the evaluation ceiling

The saved baseline and transformer both have 100% accuracy on the supplied 1,200 test rows. Four of the eight training classes are absent. Under the declared eight-class macro-F1 metric with `zero_division=0`, both achieve the ceiling of 0.50; available improvement is zero, versus the required eight points. Changing to present-class F1 would make both scores 1.00 and still leave zero improvement.

Reproduce: `python scripts/topic_target_audit.py`.

Proposed alternative: retain this documented ceiling result and use a new independently labelled, unseen evaluation set as a possible replacement protocol.

### Lab 4: LOCATION recall already 100%

Original and consistently D3-trained NER both achieve 100% LOCATION recall on the same 935 validation examples. Available improvement above the original model is zero, versus the requested four points. This is validation evidence, not a new independent test result.

Reproduce: `python scripts/compare_ner_d3.py`.

Proposed alternative: retain the measured zero gain and the original pipeline. An independently annotated NER set remains a possible replacement protocol. The degraded inference-only D3 experiment is not used as a manufactured gain.

### Lab 5: retrieval and relevance coverage both need attention

Reproduce: `python scripts/retrieval_label_audit.py`.

The audit verifies source hashes and saved query/label correspondence. It makes no model calls and preserves all original labels and results. Full per-query evidence is in `artifacts/lab5/relevance_audit.json`.

Current official scores: recall@10 0.023077 and MRR@10 0.026484. Targets remain 0.80 and 0.70.

Only nine of 130 answerable queries have any labelled relevant ID in their 50 candidates. Even an oracle ordering of these candidates could reach only recall@10 0.025641 and MRR@10 0.069231. These are analytical upper bounds for the existing candidate pools, not measured model improvements or bounds on other retrievers.

For Q-001, the top two reranked cases (CASE-002537 and CASE-008921) match the query text exactly, but are absent from the relevant-ID list. Of 130 queries, 103 have an exact text match in the corpus, 98 retrieve one in the top ten, and eight have an exact text match in the relevant-ID list. Identical complaint text can describe distinct cases and resolutions, so this is evidence requiring relevance review, not automatic permission to relabel cases or a substitute success metric.

Technical next step: inspect candidate diversity and encoder/index correctness, then test any retrieval changes on a separate development set. A larger or more diverse candidate pool may help coverage but does not guarantee target scores. Do not use the evaluation's gold IDs to select or promote candidates.

Proposed alternative: retain the retrieval integrity/relevance audit and the predeclared hybrid development probe. The original-score report is preserved. Independent held-out judgments remain a possible follow-up.

### Lab 6: individual review complete

The reviewer completed and confirmed all 120 sampled entries. The 45 grouped decisions and Arabic notes are retained as a compact summary. See [review method](LAB6_REVIEW_METHOD.md), [completed worksheet](LAB6_SHORT_REVIEW.md), and the [individual review record](../artifacts/lab6/human_error_review.json).

Individual-entry confirmations remain separate from grouped decisions. The original texts, labels and predictions are unchanged.

## Suggested project note

> We reproduced the remaining issues and attached the evidence. Labs 3 and 4 reach the current evaluation ceilings, leaving no positive headroom on those sets. Lab 5 retains the original exact-ID scores alongside the retrieval audit and hybrid probe. Lab 6 has been individually reviewed 120/120. The original requirements, datasets and scores remain unchanged.

The Lab 6 individual review is complete separately. Labs 3–5 retain their additional evidence and original measurements.
