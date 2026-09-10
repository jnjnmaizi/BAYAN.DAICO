import pytest

from bayan.search.hybrid import reciprocal_rank_fusion


def test_fusion_rewards_agreement_without_raw_score_scales():
    ranking, scores = reciprocal_rank_fusion([['a', 'b'], ['b', 'c']])
    assert ranking == ['b', 'a', 'c']
    assert scores['b'] == pytest.approx(1/62 + 1/61)
    assert scores['a'] == pytest.approx(1/61)


def test_ties_use_input_order_not_case_id_order():
    ranking, _ = reciprocal_rank_fusion([['z'], ['a']])
    assert ranking == ['z', 'a']


def test_invalid_rankings_cannot_double_count_one_candidate():
    with pytest.raises(ValueError, match='unique'):
        reciprocal_rank_fusion([['a', 'a']])
    with pytest.raises(ValueError, match='positive'):
        reciprocal_rank_fusion([['a']], constant=0)
    assert reciprocal_rank_fusion([]) == ([], {})
