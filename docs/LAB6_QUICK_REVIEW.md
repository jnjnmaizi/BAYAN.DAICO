# Lab 6 — Error analysis summary

**Review status:** 45 grouped assessments completed, covering all 120 sampled entries. Each assessment includes a decision and Arabic reviewer notes. The repository owner reports instructor acceptance of the grouped method.

[Completed worksheet](LAB6_SHORT_REVIEW.md) · [Review method and provenance](LAB6_REVIEW_METHOD.md) · [Evaluation report](../EVALUATION_REPORT.md)

## Findings

All sampled errors are labelled **parks** and predicted as **roads**. The sample contains 45 distinct text/label/prediction combinations and three recurring scenarios.

| Scenario | Sampled entries | Interpretation |
|---|---:|---|
| Playground maintenance | 42 | The request concerns play equipment inside a park. |
| Park accessibility | 46 | The request concerns a walkway inside a park, including wheelchair access. |
| Park irrigation | 32 | The request concerns irrigation inside a named park. Ownership could differ under another taxonomy; the supplied label is parks. |

Every text explicitly mentions a park. Street names within a park location do not establish that the requested service concerns roads. Thirteen examples contain spelling or elongation variants; their causal contribution has not been tested.

The supplied prediction file contains 300 errors, all parks → roads. These are course-provided predictions, distinct from the saved predictions of the trained topic classifier. The file alone does not establish whether the underlying cause is model behaviour, label mapping or data construction.

## Prioritised follow-up

1. Verify the source of the prediction file and its label-ID mapping.
2. Use paired park/road examples to test context interpretation, including road words in place names.
3. Compare clean and noisy versions of the same complaints before changing preprocessing.

Expected gains from these interventions remain unmeasured. Replacing the 120 sampled wrong predictions with gold labels would add 8.42 macro-F1 points; replacing all 300 would add 16.67 points. These are accounting ceilings, not results from implemented model changes.

## Records

The [grouped review record](../artifacts/lab6/human_group_review.json) preserves the submitted decisions, notes, group membership and reported acceptance. Individual-entry confirmations remain separate. Original texts, labels and predictions are unchanged.
