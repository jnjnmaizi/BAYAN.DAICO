"""Experimental text-group candidates; original case records remain intact."""
from collections import defaultdict

from bayan.preprocessing.core import preprocess


def distinct_candidates(rows, ranked_ids, ranked_scores, limit=50):
    """Select distinct normalized texts, retaining every member's full metadata.

    ranked_ids are index row positions, never evaluation relevance labels.
    The first occurrence determines group rank; tied cases stay in one group.
    """
    if limit < 1 or len(ranked_ids) != len(ranked_scores):
        raise ValueError('Require a positive limit and aligned ranks/scores')
    members = defaultdict(list)
    texts = []
    seen_ids = set()
    for row in rows:
        if row['case_id'] in seen_ids:
            raise ValueError('Corpus contains duplicate case IDs')
        seen_ids.add(row['case_id'])
        text = preprocess(row['case_text'])
        texts.append(text)
        members[text].append(dict(row))
    selected = []
    seen_texts = set()
    for index, score in zip(ranked_ids, ranked_scores):
        index = int(index)
        if not 0 <= index < len(rows):
            raise ValueError('Candidate index is outside corpus')
        text = texts[index]
        if text in seen_texts:
            continue
        seen_texts.add(text)
        selected.append({'case_text': text, 'bi_score': float(score), 'cases': members[text]})
        if len(selected) == limit:
            break
    return selected
