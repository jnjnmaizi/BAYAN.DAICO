# Model Card — Jana Alhumaizi NLP — ner

## Intended use
Entity extraction for course demonstration; human review required for operational decisions.

## Artefact / data versions
- Model/checkpoint: xlm-roberta-base @ e73636d4f797dec63c3081bb6ed5c7b0bb3f2089
- Preprocessing version: 1.2.0
- Data version/snapshot: e76f187728fd408954364c1dc9822af383db8c2e0bbdaf72e8df9b35d6a1752e

## Metrics
```json
{
  "entity_f1": 1.0,
  "entity_precision": 1.0,
  "entity_recall": 1.0
}
```

## Slice metrics
See [evaluation report](../EVALUATION_REPORT.md) and the corresponding Lab 3/4/6 evidence. Missing dialects and entity types are not assigned zero performance.

## Behavioural tests
Classification probes do not apply to NER. Label-alignment contracts and paired quantization validation are recorded in the test suite and Lab 7 evidence.

## Known limitations
Only four validation templates and two frozen-test templates; no ORGANISATION labels and fixed dates. High entity-F1 does not imply real-world entity coverage.

## Contact / owner
Repository maintainer; educational project.
