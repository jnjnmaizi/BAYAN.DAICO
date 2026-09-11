# Secondary external-data results — 2026-09-11

These results come from the independent external-data track. They do not
rewrite the official Bayan datasets, labels, requirements or Lab 3–5 scores.

## Dataset coverage

| Dataset | Loaded data | Relevant coverage |
|---|---:|---|
| ArBNTopic | 15,827 train / 2,374 validation / 1,583 test | 14 Arabic topic classes |
| Alyah | 1,173 test questions | 7 Emirati dialect categories |
| IAHLT Arabic NER | 2,000 streamed rows | 1,179 location-like spans: 1,055 GPE and 124 LOC |
| ArabicRAGB | 2,000 streamed rows / 1,603 unique passages | 593 Gulf, 605 Egyptian, 115 Levantine, 96 Maghrebi and 591 MSA queries |

## Independent topic result

The separate character TF-IDF plus logistic-regression baseline trained on
ArBNTopic achieved:

| Metric | Result |
|---|---:|
| Test rows | 1,583 |
| Labels | 14 |
| Accuracy | 0.740366 |
| Macro-F1 | 0.719714 |

This demonstrates a usable, fully covered Arabic topic benchmark. It is not a
replacement for Bayan's eight-label frozen-test score because the label set
and task distribution are different.

## Independent retrieval result

The character TF-IDF baseline was evaluated on 500 ArabicRAGB training records,
with one supplied positive passage per query:

| Metric | Result |
|---|---:|
| Sample queries | 500 |
| Unique passages | 478 |
| Recall@10 | 0.946000 |
| MRR@10 | 0.829494 |

This external result exceeds the Lab 5 target thresholds under a clean
query-positive-passage definition. It does not replace the official Bayan
exact-case-ID score, which remains preserved separately.

## What this proves

- The secondary evaluation has complete topic coverage for an Arabic topic
  benchmark.
- The secondary evaluation includes a real Emirati/Gulf dialect benchmark.
- The NER audit has substantial independent location coverage.
- Retrieval metrics can exceed the requested thresholds when relevance is
  defined by explicit positive passages rather than the supplied cyclic case
  IDs.

The external benchmark results are supporting evidence. They must remain
labelled as secondary unless the assessment protocol explicitly permits them as
replacement test sets.
