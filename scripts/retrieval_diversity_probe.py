"""Compare candidate diversity on new unlabelled development queries.

Uses the existing index and checkpoints. Original evaluation is not rerun.
This checks functionality and diversity, not relevance accuracy or a lab pass.
"""
import json
import time

import torch

from bayan.models.training import ROOT, write_json, sha256
from bayan.preprocessing.core import preprocess
from bayan.search.grouped import distinct_candidates
from bayan.search.service import CaseSearch

QUERIES = [
    'أبحث عن بلاغ سابق عن حفرة بجوار مدخل الحي تعيق مرور السيارات',
    'The wheelchair path inside our neighbourhood park needs repair.',
    'هل توجد حالات مشابهة لتعطل إنارة الشارع قرب المنازل؟',
]


def main():
    output = ROOT / 'artifacts/lab5/diversity_development_probe.json'
    if output.exists():
        print(output.read_text())
        return
    print('Loading existing checkpoints on CPU', flush=True)
    service = CaseSearch(str(ROOT / 'artifacts/search/cases'), device='cpu')
    results = []
    for query in QUERIES:
        start = time.perf_counter()
        text = preprocess(query)
        vector = service.encoder.encode([text])
        scores, ids = service.index.search(vector, service.index.ntotal)
        baseline = [service.rows[int(i)] for i in ids[0][:50]]
        groups = distinct_candidates(service.rows, ids[0], scores[0], limit=50)
        values = []
        with torch.inference_mode():
            for offset in range(0, len(groups), 32):
                batch_groups = groups[offset:offset+32]
                batch = service.tokenizer([text]*len(batch_groups), [r['case_text'] for r in batch_groups],
                                          padding=True, truncation=True, max_length=256, return_tensors='pt')
                values.extend(torch.sigmoid(service.reranker(**batch).logits.flatten()).tolist())
        ranked = sorted([{**row, 'score': score} for row, score in zip(groups, values)],
                        key=lambda row: -row['score'])
        result = {'query': query,
                  'original_top50_distinct_texts': len({preprocess(r['case_text']) for r in baseline}),
                  'grouped_candidate_distinct_texts': len(groups),
                  'top10_group_distinct_texts': len({r['case_text'] for r in ranked[:10]}),
                  'case_records_represented_in_50_groups': sum(len(g['cases']) for g in groups),
                  'elapsed_ms_including_full_index_search_grouping_and_reranking': 1000*(time.perf_counter()-start),
                  'top5_groups': [{'case_text': r['case_text'], 'score': r['score'],
                                   'member_case_ids': [case['case_id'] for case in r['cases']]} for r in ranked[:5]]}
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != 'top5_groups'}, ensure_ascii=False), flush=True)
    report = {'scope': 'Three newly authored unlabelled development queries; descriptive probe only',
              'device': 'cpu', 'new_training': False, 'original_evaluation_modified': False,
              'relevance_labels_used_to_select_candidates': False,
              'official_recall_mrr_targets_reassessed': False,
              'index_sha256': service.manifest['index_sha256'],
              'helper_sha256': sha256(ROOT/'src/bayan/search/grouped.py'),
              'protocol': 'Search all index positions once; take first 50 distinct normalized text groups, retain all member case records, rerank each group once. No threshold is calibrated or applied.',
              'limitations': 'No held-out relevance judgments; broader candidate coverage does not establish semantic quality or exact-ID target attainment. Group scores describe shared text, not verification of each resolution. Default API behavior is unchanged.',
              'queries': results}
    write_json(output, report)
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
