"""Assistant review must preserve source evidence and never forge human sign-off."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prepare_assistant_review',ROOT/'scripts/prepare_assistant_review.py')
review_module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(review_module)


def test_review_covers_original_ids_without_mutating_human_answers():
    rows=json.loads((ROOT/'artifacts/lab6/human_error_review.json').read_text())
    rows[0]['human_confirmed']=True
    rows[0]['category']='Existing human assessment'
    original=copy.deepcopy(rows)
    output=review_module.build_review(rows)
    assert rows==original
    assert [r['feedback_id'] for r in output]==[r['feedback_id'] for r in rows]
    assert len({r['feedback_id'] for r in output})==120
    assert all(r['human_confirmed'] is False and r['reviewer_type']=='assistant' for r in output)
    assert all(r['evidence_phrase'] in r['text'] for r in output)


@pytest.mark.parametrize('change',['text','label'])
def test_stale_annotations_rejected_when_source_evidence_changes(change):
    rows=json.loads((ROOT/'artifacts/lab6/human_error_review.json').read_text())
    if change=='text':rows[0]['text']='A billing complaint'
    else:rows[0]['y_true']='billing'
    with pytest.raises(ValueError,match='no longer matches source'):
        review_module.build_review(rows)
