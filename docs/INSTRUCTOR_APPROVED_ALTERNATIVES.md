# Instructor-approved alternatives

**Recorded: 2026-09-11**

The repository owner reports that the instructor accepted the documented alternative evidence for Labs 3–5. This closes the completion decision without changing the course requirements, source data, gold labels, or original measurements.

| Lab | Accepted completion evidence | Raw result retained |
|---|---|---|
| 3 | Frozen-test feasibility audit: both models are already at the fixed-label ceiling, so the requested gain has no available headroom. | Accuracy 1.0000; macro-F1 0.5000 for both; delta 0.00 points. |
| 4 | Paired NER comparison: the original and consistently D3-trained pipelines are already tied at the measured ceiling. | LOCATION recall 1.0000 for both; delta 0.00 points. |
| 5 | Retrieval integrity/relevance audit plus the predeclared hybrid development probe. | Official exact-ID reranked recall@10 0.023077 and MRR@10 0.026484; development probe results remain separately identified. |

The completion label used in project status is **satisfied by instructor-approved alternative**. The original metric files continue to report whether the numeric targets themselves were reached; no target, label, or dataset was rewritten to manufacture a pass.

Evidence:

- [Acceptance record](../artifacts/instructor_approved_alternatives.json)
- [Labs 3–5 follow-up](LABS_3_5_FOLLOWUP.md)
- [Lab 3 audit](../artifacts/lab3/topic_target_audit.json)
- [Lab 4 comparison](../artifacts/lab4/ner_d3_comparison.json)
- [Lab 5 retrieval report](../artifacts/lab5/retrieval.json)
