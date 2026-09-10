import copy

import pytest

from bayan.search.grouped import distinct_candidates


def test_duplicates_do_not_consume_slots_and_all_case_details_are_retained():
    rows = [
        {'case_id': 'A', 'case_text': 'park repair', 'resolution': 'pending'},
        {'case_id': 'B', 'case_text': 'park repair', 'resolution': 'repaired'},
        {'case_id': 'C', 'case_text': 'road repair', 'resolution': 'scheduled'},
        {'case_id': 'D', 'case_text': 'park repair', 'resolution': 'inspected'},
    ]
    before = copy.deepcopy(rows)
    result = distinct_candidates(rows, [1, 0, 2, 3], [.9, .9, .8, .9], limit=2)
    assert [r['case_text'] for r in result] == ['park repair', 'road repair']
    assert result[0]['cases'] == [rows[0], rows[1], rows[3]]
    assert result[1]['cases'] == [rows[2]]
    result[0]['cases'][0]['resolution'] = 'changed in response'
    assert rows == before


def test_normalization_equivalent_texts_share_a_group():
    rows = [{'case_id': 'A', 'case_text': 'park   repair'},
            {'case_id': 'B', 'case_text': 'park repair'},
            {'case_id': 'C', 'case_text': 'water supply'}]
    result = distinct_candidates(rows, [0, 1, 2], [.9, .9, .7], limit=50)
    assert len(result) == 2
    assert len(result[0]['cases']) == 2


def test_invalid_case_identity_and_ranking_are_rejected():
    rows = [{'case_id': 'A', 'case_text': 'park'}]
    with pytest.raises(ValueError, match='duplicate case IDs'):
        distinct_candidates(rows+rows, [0], [.9])
    with pytest.raises(ValueError, match='outside corpus'):
        distinct_candidates(rows, [-1], [.9])
    with pytest.raises(ValueError, match='aligned'):
        distinct_candidates(rows, [0], [])
