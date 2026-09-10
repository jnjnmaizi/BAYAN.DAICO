"""Audit saved retrieval evidence without inference or changing relevance labels."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    corpus_path = root / 'data/search/bayan_cases.csv'
    queries_path = root / 'data/search/bayan_queries.jsonl'
    results_path = root / 'artifacts/lab5/query_results.json'
    out = root / 'artifacts/lab5'
    protocol = json.loads((out / 'protocol.json').read_text())
    manifest = json.loads((out / 'index_manifest.json').read_text())
    if digest(queries_path) != protocol['query_sha256']:
        raise ValueError('Query data differs from saved evaluation protocol')
    if digest(corpus_path) != manifest['data_sha256']:
        raise ValueError('Corpus differs from saved index manifest')
    with corpus_path.open(encoding='utf-8-sig', newline='') as handle:
        corpus = list(csv.DictReader(handle))
    cases = {r['case_id']: r for r in corpus}
    if len(cases) != len(corpus):
        raise ValueError('Duplicate case IDs')
    queries = [json.loads(line) for line in queries_path.read_text().splitlines() if line.strip()]
    saved = json.loads(results_path.read_text())
    if len(queries) != len(saved):
        raise ValueError('Query/result counts differ')
    exact = defaultdict(set)
    for row in corpus:
        exact[row['case_text']].add(row['case_id'])
    details = []
    for query, row in zip(queries, saved):
        if any(row.get(key) != value for key, value in query.items()):
            raise ValueError('Saved query fields or relevance labels differ from source')
        for key in ['bi', 'reranked']:
            if len(row[key]) != len(set(row[key])) or set(row[key]) - cases.keys():
                raise ValueError('Unknown or repeated retrieved IDs')
        if set(row['reranked']) != set(row['bi']):
            raise ValueError('Reranking must preserve the candidate set')
        gold = set(query['relevant_case_ids'])
        if gold - cases.keys():
            raise ValueError('Unknown relevant ID')
        if query['no_answer']:
            continue
        if not gold:
            raise ValueError('Answerable query has no relevant IDs')
        candidates = set(row['bi'])
        top10 = row['reranked'][:10]
        available = len(gold & candidates)
        details.append({
            'query_id': query['query_id'], 'query': query['query'],
            'relevant_case_ids': query['relevant_case_ids'],
            'candidate_count': len(candidates),
            'gold_in_candidates': available,
            'candidate_recall': available / len(gold),
            'oracle_recall_at_10_for_saved_candidates': min(10, available) / len(gold),
            'recall_at_10': len(gold & set(top10)) / len(gold),
            'mrr_at_10': next((1 / (i + 1) for i, case_id in enumerate(top10) if case_id in gold), 0),
            'exact_text_in_corpus': bool(exact[query['query']]),
            'exact_text_in_gold': bool(exact[query['query']] & gold),
            'exact_text_in_top10': bool(exact[query['query']] & set(top10)),
            'expected_cases': [cases[case_id] for case_id in query['relevant_case_ids']],
            'retrieved_top3': [cases[case_id] for case_id in top10[:3]],
        })
    if not details:
        raise ValueError('No answerable queries')
    n = len(details)
    report = {
        'scope': 'Post-hoc diagnostics on saved evaluation; not a new evaluation or replacement metric',
        'new_inference': False, 'labels_modified': False,
        'source_sha256': {str(p.relative_to(root)): digest(p) for p in [corpus_path, queries_path, results_path]},
        'answerable_queries': n,
        'candidate_counts': sorted({r['candidate_count'] for r in details}),
        'queries_with_gold_in_candidates': sum(r['gold_in_candidates'] > 0 for r in details),
        'saved_reranked_recall_at_10': sum(r['recall_at_10'] for r in details) / n,
        'saved_reranked_mrr_at_10': sum(r['mrr_at_10'] for r in details) / n,
        'oracle_recall_at_10_for_saved_candidates': sum(r['oracle_recall_at_10_for_saved_candidates'] for r in details) / n,
        'oracle_mrr_at_10_for_saved_candidates': sum(r['gold_in_candidates'] > 0 for r in details) / n,
        **{key: sum(r[key] for r in details) for key in ['exact_text_in_corpus', 'exact_text_in_gold', 'exact_text_in_top10']},
        'interpretation': [
            'Oracle values assume perfect ordering of available gold IDs; they are bounds for the saved candidate pools only, not model scores.',
            'A better reranker alone cannot meet the targets with these saved candidates.',
            'Exact text overlap is diagnostic evidence of incomplete relevance coverage; identical complaints can refer to distinct cases and resolutions.',
            'This audit does not prove that target scores are impossible with a different retriever or establish general search quality.',
            'Do not select case IDs from evaluation labels, alter gold labels, or replace official metrics with exact-text counts.',
        ],
        'queries': details,
    }
    return report


def main():
    report = audit(ROOT)
    output = ROOT / 'artifacts/lab5/relevance_audit.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'queries'}, indent=2))
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
