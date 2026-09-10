# Model Card — Jana Alhumaizi NLP — topic

## Intended use
Citizen-feedback topic classification for course demonstration; human review required for operational decisions.

## Artefact / data versions
- Model/checkpoint: xlm-roberta-base @ e73636d4f797dec63c3081bb6ed5c7b0bb3f2089
- Preprocessing version: 1.2.0
- Data version/snapshot: 4a13286ed3ad850dc12a12fa69c6621a0a5126711d50c47139c8ef1b55b2523c

## Metrics
```json
{
  "point": 1.0,
  "low": 1.0,
  "high": 1.0,
  "n": 2400,
  "groups": 960,
  "small_slice": false
}
```

## Slice metrics
See [evaluation report](../EVALUATION_REPORT.md) and the corresponding Lab 3/4/6 evidence. Missing dialects and entity types are not assigned zero performance.

## Behavioural tests
Invariance 200/200; bilingual MFT 15/16. Sentiment direction uses a separate baseline.

## Known limitations
Synthetic repeated feedback; no Gulf validation and no Arabic frozen test. Eight-label macro-F1 can be capped by missing classes.

## Contact / owner
Repository maintainer; educational project.
