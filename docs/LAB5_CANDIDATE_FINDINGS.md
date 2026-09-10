# Lab 5 — Candidate diagnosis and development probe

The existing index passed integrity checks. Duplicate text crowds the current candidate pools. An experimental grouped candidate path reduces repetition, but relevance remains unresolved and it has not replaced the serving API.

## Verified diagnosis

Run `python scripts/retrieval_candidate_audit.py` to reproduce the checks from saved evidence, without query inference or training.

- All 20,000 corpus rows match index metadata, including order.
- Every stored index vector matches its normalized saved raw vector exactly; norms are within float32 tolerance of one.
- Identical normalized texts have identical saved vectors.
- The corpus contains 5,401 unique normalized texts.
- Among 130 answerable evaluation queries, 38 have just one distinct text in their 50 candidates. The mean is 10.49 distinct texts per 50 candidates.
- The reranked top ten averages 3.34 distinct texts; 44 queries have only one.

Evidence: [candidate audit](../artifacts/lab5/candidate_audit.json). These checks detect neither index corruption nor a corpus/metadata mismatch; they do not verify general model correctness.

## Experimental change

[`distinct_candidates`](../src/bayan/search/grouped.py) selects groups by normalized case text in vector-rank order. Every group retains all original case records, including distinct IDs, resolutions and dates. It never consults relevant-ID labels. A cross-encoder scores each group text once.

This is an experimental group response format, not a drop-in replacement for the original exact-case-ID evaluation. The original API, calibrated threshold, corpus, and official metrics are unchanged. Full-index retrieval and grouping add processing work; speed claims from the existing API benchmarks do not apply to this prototype.

## Three new development queries

Run `python scripts/retrieval_diversity_probe.py`. It displays the saved report if one exists; otherwise it loads the existing checkpoints on CPU and runs the three new, unlabelled queries. No training or Colab is required when the local model artifacts are available.

| Query intent | Distinct texts in original 50 candidates | Experimental candidate groups | Distinct top-ten groups | Case records retained in candidate groups |
|---|---:|---:|---:|---:|
| Arabic pothole near neighbourhood entrance | 25 | 50 | 10 | 133 |
| English wheelchair path inside a park | 7 | 50 | 10 | 464 |
| Arabic residential street lighting | 4 | 50 | 10 | 516 |

Evidence: [development probe](../artifacts/lab5/diversity_development_probe.json).

**Relevance counterexample:** the English wheelchair-access query ranks Arabic road-maintenance complaints first, ahead of park context. Increasing diversity alone has therefore not solved the semantic ranking issue. The pothole and lighting examples contain top results consistent with their broad topic, but that observation is a qualitative inspection, not an independently judged accuracy score.

The probe has no relevance labels, no independent held-out evaluation and no acceptance threshold. Do not treat 50 distinct candidates or ten distinct groups as a recall/MRR pass. The original official scores remain recall@10 0.023077 and MRR@10 0.026484.

## Validation and next step

Fourteen focused tests passed, including preservation of duplicate case metadata, unchanged source records, normalization-based grouping, invalid IDs, API contracts, and existing serving boundaries. Original test expectations were not changed.

Next: compare the bi-encoder and cross-encoder ordering on independently judged development examples, especially park accessibility versus roads and Arabic/English pairs. Keep grouped candidates experimental until relevance and latency are assessed under an agreed evaluation protocol.
