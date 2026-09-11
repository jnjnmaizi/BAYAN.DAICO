"""Supplementary retrieval quality probe using exact-text duplicates as relevance.

Read-only: reuses the already-saved query_results.json (no new inference) and
the frozen corpus/query files (hashed, not modified). Does not touch or
replace the official recall@10/MRR@10 metric or relevant_case_ids.

Rationale: retrieval_label_audit.py and relevant_id_construction_audit.py
show the supplied relevant_case_ids are assigned by a topic-index cycle
unrelated to query content, so they almost never reward finding a verbatim
duplicate of the query among the cases. A verbatim duplicate case is,
however, an extremely conservative and defensible notion of "relevant" on
its own terms. This script measures recall@10/MRR@10 against that
alternative definition, for the subset of queries where at least one
verbatim duplicate exists, purely as supplementary evidence about retriever
behaviour -- not a substitute for the official metric or for a separately
agreed relevance definition.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(root):
    corpus_path = root / 'data/search/bayan_cases.csv'
    queries_path = root / 'data/search/bayan_queries.jsonl'
    results_path = root / 'artifacts/lab5/query_results.json'

    with corpus_path.open(encoding='utf-8-sig', newline='') as handle:
        corpus = list(csv.DictReader(handle))
    exact = {}
    for row in corpus:
        exact.setdefault(row['case_text'], set()).add(row['case_id'])

    saved = json.loads(results_path.read_text())

    details = []
    for row in saved:
        if row['no_answer']:
            continue
        alt_gold = exact.get(row['query'], set())
        if not alt_gold:
            continue
        top10 = row['reranked'][:10]
        hit_rank = next((i for i, cid in enumerate(top10) if cid in alt_gold), None)
        details.append({
            'query_id': row['query_id'],
            'official_relevant_case_ids': row['relevant_case_ids'],
            'exact_duplicate_case_ids': sorted(alt_gold),
            'in_top10': hit_rank is not None,
            'rank_if_found': (hit_rank + 1) if hit_rank is not None else None,
        })

    n = len(details)
    recall_at_10 = sum(d['in_top10'] for d in details) / n if n else None
    mrr_at_10 = sum(1 / d['rank_if_found'] for d in details if d['in_top10']) / n if n else None

    report = {
        'scope': 'Supplementary diagnostic only; does not replace or modify the official metric or relevant_case_ids',
        'new_inference': False, 'labels_modified': False,
        'source_sha256': {
            str(corpus_path.relative_to(root)): digest(corpus_path),
            str(queries_path.relative_to(root)): digest(queries_path),
            str(results_path.relative_to(root)): digest(results_path),
        },
        'relevance_definition': 'A case whose case_text is byte-identical to the query text',
        'queries_with_a_verbatim_duplicate_in_corpus': n,
        'recall_at_10_under_exact_duplicate_definition': recall_at_10,
        'mrr_at_10_under_exact_duplicate_definition': mrr_at_10,
        'for_contrast_official_recall_at_10': 0.023076923076923078,
        'for_contrast_official_mrr_at_10': 0.026483516483516486,
        'interpretation': [
            'This measures whether the saved reranked pipeline finds a verbatim duplicate of the '
            'query text among the top 10 cases, using the duplicate case IDs (not relevant_case_ids) '
            'as the relevance set.',
            'It is evidence about retriever/reranker behaviour under one conservative relevance '
            'notion, not proof of correctness under the official one, and it covers only the subset '
            'of queries that happen to have a verbatim duplicate in the corpus.',
            'It should be presented alongside relevant_id_construction_audit.json, not used to replace '
            'the official recall@10/MRR@10 in the evaluation report.',
        ],
        'queries': details,
    }
    return report


def main():
    report = probe(ROOT)
    output = ROOT / 'artifacts/lab5/exact_match_relevance_probe.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'queries'}, indent=2, ensure_ascii=False))
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
