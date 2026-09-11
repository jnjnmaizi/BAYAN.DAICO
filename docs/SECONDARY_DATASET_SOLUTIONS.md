# Secondary dataset solutions

This document describes an independent evaluation track for the three targets
that the supplied Bayan test data cannot measure fairly. The original data,
labels, requirements and official result files remain unchanged.

## Why a secondary track is needed

The official Lab 3 test contains only four of the eight training classes and
the baseline is already at the fixed-label ceiling. The official Lab 4
validation has no Gulf rows, and the original LOCATION recall is already
100%. The official Lab 5 labels follow an eight-row cycle and do not reliably
identify the text matches returned by retrieval.

These are evaluation-set problems. More training on the same frozen test
cannot create missing classes, Gulf rows or corrected relevance judgments.

## Selected public datasets

| Blocked area | Dataset | Use in this project | Why it fits |
|---|---|---|---|
| Lab 3 topic coverage | [ArBNTopic](https://huggingface.co/datasets/U4RASD/ArBNTopic) | Independent Arabic topic test with 14 classes | It has a real train/dev/test structure and broad Arabic topic coverage. Its labels are not renamed to Bayan's eight labels. |
| Lab 4 Gulf coverage | [Alyah](https://huggingface.co/datasets/tiiuae/alyah-emirati-benchmark) | Emirati dialect robustness check | It is a manually curated Emirati benchmark with 1,173 test questions. It is used as Gulf evidence, not as a fake LOCATION score. |
| Lab 4 location coverage | [IAHLT Arabic NER](https://huggingface.co/datasets/iahlt/arabic_ner_mafat) | Independent Arabic LOC/GPE span check | It provides span and token NER annotations and has location-like entity labels. |
| Lab 5 retrieval | [ArabicRAGB](https://huggingface.co/datasets/HeshamHaroon/ArabicRAGB) | External Arabic retrieval benchmark | Each record supplies a query and its positive passage ID, including Gulf queries, so recall@10 and MRR@10 have explicit relevance definitions. |

The gated ArSyra Gulf dataset was not selected because its full data requires
restricted access and a separate licence. The selected track uses public
dataset cards and does not commit downloaded raw records to this repository.

## Reproduction

Install the existing project dependencies, then run:

```bash
python scripts/secondary_dataset_audit.py
python scripts/secondary_retrieval_eval.py
```

The scripts write only derived summaries to `artifacts/secondary/`. They do
not write to `data/`, overwrite official artifacts, or change the production
API.

## Interpretation

The external datasets provide valid additional evidence and remove the
coverage gaps in the original frozen evaluation. They do not retroactively
turn the original Lab 3–5 numbers into passes. A course score can use these
results only if the evaluation protocol allows an external benchmark or a
replacement test set.

## Files

- `scripts/secondary_dataset_audit.py` records split sizes, label coverage,
  dialect coverage and NER location coverage.
- `scripts/secondary_topic_eval.py` trains a transparent 14-class TF-IDF
  baseline on ArBNTopic when the dataset is available.
- `scripts/secondary_retrieval_eval.py` runs a transparent character TF-IDF
  retrieval baseline on a 500-record ArabicRAGB sample.
- `artifacts/secondary/dataset_audit.json` contains derived dataset metadata.
- `artifacts/secondary/arabicragb_retrieval.json` contains the independent
  retrieval result.
