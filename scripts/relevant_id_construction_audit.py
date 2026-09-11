"""Diagnose how the supplied relevant_case_ids were constructed.

Read-only: makes no model calls, changes no labels, and does not touch the
frozen corpus or query files beyond hashing/reading them. This complements
scripts/retrieval_label_audit.py by explaining *why* gold-in-candidate
coverage is so low: it tests whether relevant_case_ids follow a fixed
positional cycle over the topic-ordered corpus, independent of query text.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    corpus_path = root / 'data/search/bayan_cases.csv'
    queries_path = root / 'data/search/bayan_queries.jsonl'

    with corpus_path.open(encoding='utf-8-sig', newline='') as handle:
        corpus = list(csv.DictReader(handle))
    index_of = {row['case_id']: i for i, row in enumerate(corpus)}
    topic_of_row = [row['topic'] for row in corpus]
    n_topics = len(set(topic_of_row))

    queries = [json.loads(line) for line in queries_path.read_text().splitlines() if line.strip()]
    answerable = [q for q in queries if not q['no_answer']]

    per_query = []
    gap_counter = Counter()
    for q in answerable:
        idxs = sorted(index_of[c] for c in q['relevant_case_ids'])
        gaps = [b - a for a, b in zip(idxs, idxs[1:])]
        gap_counter[tuple(gaps)] += 1
        topics_hit = {topic_of_row[i] for i in idxs}
        per_query.append({
            'query_id': q['query_id'],
            'query_topic': q['topic'],
            'relevant_case_ids': q['relevant_case_ids'],
            'zero_based_indices': idxs,
            'consecutive_gaps': gaps,
            'gaps_equal_topic_count': all(g == n_topics for g in gaps),
            'all_relevant_share_query_topic': topics_hit == {q['topic']},
        })

    n = len(per_query)
    n_fixed_gap = sum(r['gaps_equal_topic_count'] for r in per_query)
    n_topic_locked = sum(r['all_relevant_share_query_topic'] for r in per_query)

    report = {
        'scope': 'Diagnoses the construction of the supplied relevant_case_ids field; makes no model calls',
        'new_inference': False, 'labels_modified': False,
        'source_sha256': {
            str(corpus_path.relative_to(root)): digest(corpus_path),
            str(queries_path.relative_to(root)): digest(queries_path),
        },
        'n_topics_in_corpus': n_topics,
        'answerable_queries': n,
        'queries_with_gap_equal_topic_count': n_fixed_gap,
        'queries_with_gap_equal_topic_count_fraction': n_fixed_gap / n,
        'queries_where_all_relevant_share_query_topic': n_topic_locked,
        'queries_where_all_relevant_share_query_topic_fraction': n_topic_locked / n,
        'observed_gap_patterns': {str(k): v for k, v in gap_counter.items()},
        'interpretation': [
            f'For every answerable query, the three relevant_case_ids are exactly {n_topics} rows apart '
            'in the untouched corpus order, i.e. the same cyclic topic-slot position repeated three times.',
            'This holds regardless of the specific wording, location, or detail in the query text.',
            'This is a structural property of how the label was assigned (position in a topic-cycled '
            'list), not a measurement of semantic or textual relevance to the query.',
            'It explains, independent of retriever/encoder/reranker choice, why gold-in-candidate '
            'coverage and recall@10/MRR@10 are near the observed floor: a correct dense/lexical match '
            '(e.g. an exact-text duplicate case) is systematically absent from the gold set unless it '
            'happens to also land on that same cyclic slot by chance.',
            'This does not prove the labels are unusable for every purpose, and it is not permission to '
            'edit relevant_case_ids, select candidates using them, or substitute this diagnostic for the '
            'official metric. It is evidence for documenting how relevance should be '
            'defined for this dataset.',
        ],
        'queries': per_query,
    }
    return report


def main():
    report = audit(ROOT)
    output = ROOT / 'artifacts/lab5/relevant_id_construction_audit.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'queries'}, indent=2, ensure_ascii=False))
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
