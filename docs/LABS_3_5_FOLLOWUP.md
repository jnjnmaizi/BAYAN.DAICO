# Labs 3–5 — Follow-up implementation and verified results

## Completion status

Additional evidence for Labs 3–5 is documented. The original numeric targets, source data, labels, and measured artifacts remain unchanged. See the [evidence record](../artifacts/alternative_evidence.json).

## Lab 3: saved evaluation rechecked

The TF-IDF baseline learns its vocabulary and classifier from training data. The transformer uses the same task-label vocabulary and supplied citizen-disjoint split. Saved evaluation data hashes and label counts pass the existing audit.

The frozen test contains four of eight task classes. Both baseline and transformer have 100% accuracy and fixed-eight-label macro-F1 0.50. With absent-class F1 set to zero, 0.50 is the ceiling; available improvement is zero versus a required eight points. Switching to present-class F1 gives both models 1.00 and still no improvement.

This follow-up reuses saved evidence. It does not retrain or rerun frozen-test inference. No correctable data-hash or label-vocabulary mismatch was found by these checks. A separate evaluation protocol and independently labelled data would be needed to measure performance beyond the current ceiling. New data cannot guarantee a positive gain.

## Lab 4: paired comparison rechecked

The original and consistently D3-trained NER artifacts have matching data hashes and identical validation IDs, original words, and gold labels. Both yield 100% LOCATION recall on 935 validation examples. The requested four-point improvement has zero headroom. The inference-only mismatched-segmentation ablation remains a separate experiment, not a substitute baseline.

Evidence for both checks: [remaining-target recheck](../artifacts/lab3/remaining_targets_recheck.json). The current public course revision remains `7949de02de71cd3ae644cd89f766dfc73aa9b7f4` at the follow-up fetch.

## Lab 5: hybrid candidate ranking implemented

The preceding [candidate audit](LAB5_CANDIDATE_FINDINGS.md) found duplicate crowding and a wheelchair-access ranking failure. A new experimental path combines:

1. Dense retrieval over distinct normalized case-text groups.
2. TF-IDF word unigram/bigram retrieval, fitted only to corpus texts.
3. Cross-encoder ranking of their candidate union.
4. Equal-weight reciprocal-rank fusion (RRF), with fixed constant 60, across the three lists.

RRF combines ranks instead of treating unrelated score scales as probabilities. See [the RRF formula](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion) and [TF-IDF vectorizer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

Each candidate group retains every original case ID and its metadata. Neither topic labels nor relevant-case labels are used as ranking features. The implementation does not alter the default API, original evaluation, course requirements or data. RRF scores are not compatible with the existing no-answer threshold; there is no new threshold calibration in this experiment.

### Predeclared development probe

The [protocol](../artifacts/lab5/hybrid_development_protocol.json) was saved before inference. It uses three previously inspected development queries and sixteen new assistant-authored prompts spanning all eight broad service topics in Arabic and English. Expected broad topics are assistant-authored diagnostics, not independent case-level relevance judgments.

| Method | Top result agrees with expected broad topic |
|---|---:|
| Dense retrieval alone | 18/19 |
| Lexical retrieval alone | 15/19 |
| Original grouped cross-encoder path | 17/19 |
| Cross-encoder over expanded union | 17/19 |
| Hybrid rank fusion | 18/19 |

The hybrid approach recovers the wheelchair-access example: the top results are now inaccessible-path complaints instead of road-maintenance complaints. It matches dense-only performance on this broad-topic proxy; the probe does not show superiority over the dense retriever alone.

**Remaining failure:** “The water supply to our home has stopped.” still retrieves water-service login failures. Such a result matches a service name but misses the requested physical service. A more representative judged development set is needed before selecting a final method.

This is not recall@10, MRR@10, an independent held-out test, or proof of full relevance. Original course scores remain recall@10 0.023077 and MRR@10 0.026484, below their numeric targets. Synthetic duplication and sparse exact-ID labels remain a separate evaluation issue. The probe is supporting analysis and is not relabelled as the official exact-ID score.

### Relevance-construction audit

The [dataset audit](LAB5_DATASET_AUDIT.md) adds a read-only check of the supplied labels. For all 130 answerable queries, the three official relevant IDs are exactly eight corpus rows apart, matching the repeating eight-topic cycle. This is a structural property of the supplied label construction and explains why a genuine text match can be counted as incorrect under the official IDs.

The saved reranker reaches recall@10 **0.951456** and MRR@10 **0.683545** for the 103 queries with a verbatim duplicate in the corpus under that conservative supplementary definition. The official exact-ID result remains unchanged; no labels or cases were modified.

### Files and reproduction

- [Hybrid implementation](../src/bayan/search/hybrid.py)
- [Development probe](../scripts/retrieval_hybrid_probe.py)
- [Saved results with top-ranked examples](../artifacts/lab5/hybrid_development_probe.json)

With the project's existing environment and local checkpoints:

```bash
python scripts/retrieval_hybrid_probe.py
```

The script prints saved results when available, avoiding repeat inference. No new package installation or Colab session is required.

### Optional follow-up after acceptance

- Labs 3–4: a new independently labelled evaluation set could measure generalisation beyond the supplied ceiling-limited split.
- Lab 5: independently judged relevance pools and a calibrated threshold could improve the research evidence. The experimental group format and uncalibrated fusion scores remain separate from the default case API.

All original result artifacts remain available. No metric target has been lowered and no failed result has been relabelled as passed.
