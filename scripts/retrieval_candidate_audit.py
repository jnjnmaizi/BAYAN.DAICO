"""Inspect saved candidates and index integrity; no training or query inference."""
import json
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from bayan.models.training import ROOT, sha256, write_json
from bayan.preprocessing.core import PREPROC_VERSION, preprocess


def audit(root):
    prefix = root / 'artifacts/search/cases'
    out = root / 'artifacts/lab5'
    manifest = json.loads(Path(f'{prefix}_manifest.json').read_text())
    corpus_path = root / 'data/search/bayan_cases.csv'
    metadata_path = Path(f'{prefix}_metadata.json')
    index_path = Path(f'{prefix}.faiss')
    for key, path in [('data_sha256', corpus_path), ('metadata_sha256', metadata_path), ('index_sha256', index_path)]:
        if sha256(path) != manifest[key]:
            raise ValueError(f'Saved index integrity check failed: {key}')
    if manifest['preproc_version'] != PREPROC_VERSION:
        raise ValueError('Index preprocessing version differs from current implementation')
    corpus = pd.read_csv(corpus_path).to_dict('records')
    metadata = json.loads(metadata_path.read_text())
    if corpus != metadata:
        raise ValueError('Corpus and index metadata differ in content or row order')
    faiss.omp_set_num_threads(1)
    index = faiss.read_index(str(index_path))
    raw_path = Path(f'{prefix}_raw.npy')
    raw = np.load(raw_path, allow_pickle=False)
    if raw.shape != (len(metadata), index.d) or index.ntotal != len(metadata):
        raise ValueError('Raw vectors, index, and metadata have different dimensions')
    if not np.isfinite(raw).all() or np.any(np.linalg.norm(raw, axis=1) == 0):
        raise ValueError('Raw vectors must be finite and nonzero')
    expected = np.ascontiguousarray(raw.copy(), dtype=np.float32)
    faiss.normalize_L2(expected)
    actual = index.reconstruct_n(0, index.ntotal)
    error = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, atol=1e-6, rtol=0):
        raise ValueError('Index vectors differ from normalized saved raw vectors')
    norms = np.linalg.norm(actual, axis=1)
    if not np.allclose(norms, 1, atol=1e-5, rtol=0):
        raise ValueError('Index contains non-unit vectors')
    texts = {r['case_id']: preprocess(r['case_text']) for r in metadata}
    if len(texts) != len(metadata):
        raise ValueError('Duplicate case IDs')
    # Verify identical inputs really have identical saved embeddings.
    representatives = {}
    duplicate_error = 0.0
    for i, row in enumerate(metadata):
        first = representatives.setdefault(texts[row['case_id']], i)
        duplicate_error = max(duplicate_error, float(np.max(np.abs(actual[i] - actual[first]))))
    protocol = json.loads((out / 'protocol.json').read_text())
    query_path = root / 'data/search/bayan_queries.jsonl'
    if sha256(query_path) != protocol['query_sha256']:
        raise ValueError('Queries differ from saved protocol')
    queries = [json.loads(line) for line in query_path.read_text().splitlines() if line.strip()]
    results_path = out / 'query_results.json'
    saved = json.loads(results_path.read_text())
    if len(queries) != len(saved):
        raise ValueError('Saved query counts differ')
    details = []
    for query, result in zip(queries, saved):
        if any(result.get(key) != value for key, value in query.items()):
            raise ValueError('Saved query or relevance labels differ')
        if query['no_answer']:
            continue
        if set(result['reranked']) != set(result['bi']):
            raise ValueError('Reranked candidates do not match bi-encoder candidates')
        if len(result['bi']) != len(set(result['bi'])):
            raise ValueError('Repeated candidate IDs')
        counts = Counter(texts[case_id] for case_id in result['bi'])
        top10 = result['reranked'][:10]
        details.append({'query_id': query['query_id'], 'candidate_count': len(result['bi']),
                        'unique_candidate_texts': len(counts),
                        'largest_identical_text_group': max(counts.values()),
                        'unique_reranked_top10_texts': len({texts[c] for c in top10})})
    if not details:
        raise ValueError('No answerable queries')
    return {
        'scope': 'Post-hoc audit of original saved results; not a new retrieval evaluation',
        'new_training_or_inference': False, 'original_labels_modified': False,
        'source_sha256': {str(path.relative_to(root)): sha256(path) for path in
                          [corpus_path, metadata_path, index_path, raw_path, query_path, results_path]},
        'integrity': {'corpus_matches_metadata_including_order': True, 'vectors': index.ntotal,
                      'dimensions': index.d, 'index_vs_normalized_raw_max_abs_error': error,
                      'unit_norm_range': [float(norms.min()), float(norms.max())],
                      'identical_input_vector_max_abs_error': duplicate_error,
                      'unique_normalized_corpus_texts': len(representatives)},
        'candidate_diversity': {
            'answerable_queries': len(details),
            'mean_distinct_texts_among_candidates': float(np.mean([r['unique_candidate_texts'] for r in details])),
            'mean_distinct_texts_in_reranked_top10': float(np.mean([r['unique_reranked_top10_texts'] for r in details])),
            'queries_with_only_one_distinct_candidate_text': sum(r['unique_candidate_texts'] == 1 for r in details),
            'queries_with_only_one_distinct_top10_text': sum(r['unique_reranked_top10_texts'] == 1 for r in details),
            'candidate_distinct_text_count_histogram': dict(sorted(Counter(r['unique_candidate_texts'] for r in details).items())),
        },
        'conclusions': [
            'No index corruption, row-order mismatch, or corpus-vector normalization failure was detected by these checks.',
            'Repeated case text crowds the fixed-size candidate pool and the displayed results.',
            'Different case IDs can have different resolutions and timestamps despite identical text; do not delete original records.',
            'Candidate diversity is a development hypothesis, not proof of better official recall or MRR.',
            'Next experiment: retrieve distinct text groups, retain all member IDs, and rerank group texts before presenting case details. Use independent development judgments and preserve original evaluation results.',
        ],
        'queries': details,
    }


def main():
    report = audit(ROOT)
    output = ROOT / 'artifacts/lab5/candidate_audit.json'
    write_json(output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ['queries', 'source_sha256']}, indent=2))
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
