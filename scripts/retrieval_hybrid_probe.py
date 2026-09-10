"""One predeclared hybrid-development experiment, not the course evaluation."""
import json
from time import perf_counter

from bayan.models.training import ROOT, write_json, sha256
from bayan.search.service import CaseSearch
from bayan.search.hybrid import HybridGroupSearch

# Broad intent labels are used only after retrieval for a diagnostic proxy.
# They are assistant-authored, not independent relevance judgments.
QUERIES = [
    ('ar', 'roads', 'أبحث عن بلاغ سابق عن حفرة بجوار مدخل الحي تعيق مرور السيارات'),
    ('en', 'parks', 'The wheelchair path inside our neighbourhood park needs repair.'),
    ('ar', 'lighting', 'هل توجد حالات مشابهة لتعطل إنارة الشارع قرب المنازل؟'),
    ('ar', 'billing', 'لدي اعتراض على مبلغ الفاتورة وأحتاج مراجعة الرسوم'),
    ('en', 'billing', 'I need help disputing an incorrect charge on my bill.'),
    ('ar', 'digital_services', 'لا يمكنني تسجيل الدخول إلى بوابة الخدمات الإلكترونية'),
    ('en', 'digital_services', 'The online service portal will not let me log in.'),
    ('ar', 'licensing', 'طلب تجديد الرخصة متأخر وأريد معرفة سبب التأخير'),
    ('en', 'licensing', 'My license renewal application has been delayed.'),
    ('ar', 'lighting', 'مصابيح الشارع لا تعمل والمنطقة مظلمة في الليل'),
    ('en', 'lighting', 'The street lights are broken and the road is dark at night.'),
    ('ar', 'parks', 'ألعاب الأطفال داخل الحديقة مكسورة وتحتاج إلى إصلاح'),
    ('en', 'parks', 'The playground equipment in the public park is broken.'),
    ('ar', 'roads', 'هناك حفرة كبيرة في الإسفلت تسبب مشكلة للسيارات'),
    ('en', 'roads', 'A large pothole in the asphalt is damaging passing cars.'),
    ('ar', 'waste', 'حاويات النفايات ممتلئة ولم يتم جمع القمامة'),
    ('en', 'waste', 'The rubbish bins are overflowing and collection has been missed.'),
    ('ar', 'water', 'انقطعت المياه عن المنزل وأحتاج بلاغاً عن انقطاع الخدمة'),
    ('en', 'water', 'The water supply to our home has stopped.'),
]


def main():
    output = ROOT/'artifacts/lab5/hybrid_development_probe.json'
    if output.exists():
        print(output.read_text())
        return
    protocol_path = ROOT/'artifacts/lab5/hybrid_development_protocol.json'
    protocol = {'purpose': 'Development investigation; no official pass claims',
                'queries': [{'lang': lang, 'expected_broad_topic': topic, 'query': query} for lang, topic, query in QUERIES],
                'query_origin': 'Three previously inspected development queries plus sixteen new assistant-authored bilingual prompts',
                'labels_origin': 'Assistant broad-intent expectations, not independent case-level relevance judgments',
                'method': 'Unique text groups; dense top50 plus positive TF-IDF word1-2 top50; cross-encoder reranks union; equal-weight RRF across dense, lexical and cross-encoder ranks with constant60',
                'proxy': 'Top-ranked group contains only cases with the expected broad topic; not recall@10 or MRR@10',
                'no_answer_threshold': 'Not calibrated; prototype must not use the existing API threshold',
                'default_api_changed': False, 'official_evaluation_rerun': False,
                'ranking_code_sha256': sha256(ROOT/'src/bayan/search/hybrid.py')}
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('An experiment protocol already exists with different settings')
    write_json(protocol_path, protocol)
    print('Protocol saved before inference. Loading local checkpoints.', flush=True)
    service = CaseSearch(str(ROOT/'artifacts/search/cases'), device='cpu')
    hybrid = HybridGroupSearch(service)
    results = []
    methods = ['dense', 'lexical', 'original_grouped_cross_encoder', 'cross_encoder', 'hybrid']
    for lang, expected, query in QUERIES:
        start = perf_counter()
        found = hybrid.rank(query)
        row = {'query': query, 'lang': lang, 'expected_broad_topic': expected,
               'candidate_groups': found['candidate_groups'], 'elapsed_ms': 1000*(perf_counter()-start), 'methods': {}}
        for name in methods:
            groups = found[name]
            match = bool(groups) and {c['topic'] for c in groups[0]['cases']} == {expected}
            row['methods'][name] = {'top1_broad_topic_match': match,
                'top3': [{'case_text': g['case_text'], 'topics': sorted({c['topic'] for c in g['cases']}),
                          'member_case_ids': [c['case_id'] for c in g['cases']],
                          'cross_encoder_score': g['cross_encoder_score'], 'rrf_score': g['rrf_score']} for g in groups[:3]]}
        results.append(row)
        print(f"{len(results)}/{len(QUERIES)} {lang} {expected}: "+str({m:row['methods'][m]['top1_broad_topic_match'] for m in methods}), flush=True)
    report = {'scope': protocol['purpose'], 'protocol_sha256': sha256(protocol_path),
              'original_query_results_sha256': sha256(ROOT/'artifacts/lab5/query_results.json'),
              'official_targets_met': 'Not reassessed',
              'broad_topic_proxy': {m:{'matches': sum(r['methods'][m]['top1_broad_topic_match'] for r in results), 'n':len(results)} for m in methods},
              'limitations': 'Development only; prompts and broad-intent expectations are assistant-authored. Topic agreement is insufficient for exact-case relevance, location fidelity, null rejection, or a course pass. No threshold or production latency claim.',
              'queries': results}
    write_json(output, report)
    print(json.dumps(report['broad_topic_proxy'], indent=2))
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
