# Lab 5 dataset and relevance audit

The official Lab 5 scores use the supplied `relevant_case_ids` field. A read-only audit checked how those IDs relate to the frozen corpus order.

For all **130 answerable queries**, the three supplied relevant IDs are exactly **8 rows apart** in the untouched 20,000-row corpus. Eight is also the number of service topics in the repeating corpus cycle. All three IDs share the query topic. This pattern is independent of the query wording, location, and details.

That structure explains why a semantically strong retrieval result can still score as incorrect under the official IDs. A verbatim duplicate can be absent from the official gold set unless it happens to occupy the same cyclic position.

The supplementary exact-duplicate probe reuses the saved `query_results.json`; it performs no new model inference. For the 103 answerable queries with a verbatim duplicate in the corpus, the saved reranked pipeline achieves:

| Relevance definition | Queries | Recall@10 | MRR@10 |
|---|---:|---:|---:|
| Official exact case IDs | 130 | 0.023077 | 0.026484 |
| Verbatim duplicate text | 103 | 0.951456 | 0.683545 |

The duplicate-text score is supplementary evidence, not a replacement for the official metric. No case rows, labels, or original result artifacts were changed.

Reproduce the audits:

```bash
python scripts/relevant_id_construction_audit.py
python scripts/exact_match_relevance_probe.py
```

Outputs:

- [Relevant-ID construction audit](../artifacts/lab5/relevant_id_construction_audit.json)
- [Exact-match relevance probe](../artifacts/lab5/exact_match_relevance_probe.json)
