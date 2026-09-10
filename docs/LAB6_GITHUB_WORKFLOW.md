# Lab 6 — Follow the course GitHub method

Verified on 2026-09-10 against the current course commit `7949de02de71cd3ae644cd89f766dfc73aa9b7f4`. The upstream method is unchanged. Course source: [Lab 6 instructions](https://github.com/AljawharaAlbahlalDev/SDA-AIE-211-Bayan-Course/tree/7949de02de71cd3ae644cd89f766dfc73aa9b7f4#lab-6--step-1-bootstrap-confidence-intervals).

| Official step | Our files / command | Current status |
|---|---|---|
| 1. Bootstrap confidence intervals | `python -m pytest tests/test_evaluation.py -q` | Implemented; six original contracts pass. |
| 2. Sliced report | `src/bayan/evaluation/slices.py`, `EVALUATION_REPORT.md` | Implemented; 16 topic slices, paired CIs and missing-Gulf limitation recorded. |
| 3. Behavioural tests | `src/bayan/evaluation/behavioural.py` | Implemented; invariance, MFT and separate sentiment-direction evidence recorded. |
| 4. Human error review | `data/eval/validation_predictions.csv`, `docs/ERROR_TAXONOMY.md` | 120 assistant annotations are supporting material; human confirmations remain pending. |
| 5. Report and model cards | `python scripts/evaluation_report.py` | Generator and three cards ready; human-review results update as decisions are saved. |

The course explicitly requires human reading for Step 4. An assistant summary is not an approved replacement. The short grouped review remains useful preparation; it does not change the official completion criteria.

## Review without editing JSON

From the project directory, with the environment activated:

```bash
python scripts/review_lab6.py --limit 5
```

The tool shows one original example, its expected/predicted labels, and any matching assistant suggestion. Read the entry, then choose:

- `a`: you read this specific example and agree with the displayed category and explanation. That explicit decision is saved with its assistant-assisted provenance.
- `e`: enter your own category and explanation.
- `s`: skip this example; it remains unconfirmed.
- `q`: stop. Previously saved decisions are retained.

There is no confirm-all command, and pressing Enter does not confirm an entry. Each confirmation is saved immediately. The next run resumes at pending entries. The tool handles display and saving; the human makes the review decision. Follow any additional instructor rules about assistant-assisted assessments or working with a classmate.

```bash
# Progress only; changes nothing:
python scripts/review_lab6.py --status

# After a review session, refresh the report:
python scripts/evaluation_report.py

# Official final commands:
python -m pytest tests/test_evaluation.py -q
python scripts/evaluation_report.py
```

Human sign-off counts only entries with a category, explanation and explicit confirmation. The confirmed histogram and report use those entries. The assistant histogram stays separately labelled. Review the proposed fixes and limitations as part of final report preparation.

The GitHub requirements, original tests, data, and existing human answers were not changed to accommodate this workflow. Source instructions are authoritative; the command-line interface is a convenience for carrying them out.
