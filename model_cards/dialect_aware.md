# Model Card — Bayan dialect_aware

## Intended use
Citizen-feedback topic classification for course demonstration; human review required for operational decisions.

## Artefact / data versions
- Model/checkpoint: CAMeL-Lab/bert-base-arabic-camelbert-da @ 231698eab9ebf0ae7b518a64277b81b2fe829f2d
- Preprocessing version: 1.2.0
- Data version/snapshot: 4a13286ed3ad850dc12a12fa69c6621a0a5126711d50c47139c8ef1b55b2523c

## Metrics
```json
{
  "point": 0.5,
  "low": 0.5,
  "high": 0.5,
  "n": 1200,
  "groups": 480,
  "small_slice": false
}
```

## Slice metrics
See [evaluation report](../EVALUATION_REPORT.md) and the corresponding Lab 3/4/6 evidence. Missing dialects and entity types are not assigned zero performance.

## Behavioural tests
Invariance 200/200; bilingual MFT 9/16. Sentiment direction uses a separate baseline.

## Known limitations
Evaluated only on MSA; no evidence of Gulf improvement. English behaviour is outside intended Arabic use.

## Contact / owner
Repository maintainer; educational project.
