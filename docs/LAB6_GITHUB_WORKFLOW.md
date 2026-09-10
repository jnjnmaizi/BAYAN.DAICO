# Lab 6 — Evaluation workflow

Course reference: revision `7949de02de71cd3ae644cd89f766dfc73aa9b7f4`. Original requirements remain in [LABS.md](LABS.md).

| Step | Evidence | Status |
|---|---|---|
| Bootstrap intervals | `tests/test_evaluation.py` | Implemented and tested. |
| Sliced report | `EVALUATION_REPORT.md` | Topic slices, paired intervals and data coverage recorded. |
| Behavioural tests | `src/bayan/evaluation/behavioural.py` | Invariance, functionality and sentiment-direction results recorded. |
| Error review | `artifacts/lab6/human_group_review.json` | 45 completed groups cover 120 entries; instructor acceptance reported by the repository owner. |
| Report and model cards | `scripts/evaluation_report.py`, `model_cards/` | Generated from saved metrics and review records. |

The [review method](LAB6_REVIEW_METHOD.md) distinguishes draft preparation, grouped decisions and individual confirmations. Use the [completed worksheet](LAB6_SHORT_REVIEW.md) to inspect the reviewer notes.

## Regenerate the report

```bash
python scripts/evaluation_report.py
python -m pytest tests/test_evaluation.py -q
```

Existing metric reports are reused. Review status is read from the individual and grouped records; regenerating the report does not add confirmations or rerun the frozen test.

## Individual review tool

For future individual review sessions:

```bash
python scripts/review_lab6.py --limit 5
```

- `a`: approve the displayed suggestion after reading the entry.
- `e`: enter a category and explanation.
- `s`: skip the entry.
- `q`: stop; saved decisions are retained.

The tool records the origin of an adopted suggestion. Each decision saves immediately, and subsequent sessions resume pending entries. Its individual-entry count is separate from the accepted grouped review.
