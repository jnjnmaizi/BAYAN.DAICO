"""Measured two-stage retrieval; supplied labels remain unchanged."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import faiss
import torch
from bayan.models.training import ROOT, write_json, sha256
from bayan.search.index import build_index
from bayan.search.service import CaseSearch
from bayan.preprocessing.core import preprocess


def metrics(rows, key):
    if not rows:
        return None
    recalls=[]; reciprocal=[]
    for row in rows:
        relevant=set(row['relevant_case_ids']); ids=row[key][:10]
        recalls.append(len(relevant.intersection(ids))/len(relevant))
        reciprocal.append(next((1/(i+1) for i,v in enumerate(ids) if v in relevant),0))
    return {'n':len(rows),'recall_at_10':float(np.mean(recalls)),'mrr_at_10':float(np.mean(reciprocal))}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--device',default='cpu'); args=parser.parse_args()
    torch.set_num_threads(4)
    prefix=ROOT/'artifacts/search/cases'
    out=ROOT/'artifacts/lab5'
    if (out/'retrieval.json').exists():
        print((out/'retrieval.json').read_text()); return
    queries=[json.loads(s) for s in (ROOT/'data/search/bayan_queries.jsonl').read_text().splitlines()]
    write_json(out/'protocol.json',{'query_sha256':sha256(ROOT/'data/search/bayan_queries.jsonl'),
        'candidates':50,'k':10,'device':args.device,'threshold_policy':'Select highest balanced empty/no-empty accuracy on supplied labelled calibration queries; report resubstitution, not independent test accuracy',
        'planted_bug':'Same raw embeddings without corpus/query L2 normalization; no artificial perturbation',
        'labels_modified':False})
    if not Path(f'{prefix}_manifest.json').exists():
        print('Building full corpus index',flush=True); build_index(prefix,device=args.device)
    service=CaseSearch(str(prefix),device=args.device)
    raw=np.load(f'{prefix}_raw.npy'); bug=faiss.IndexFlatIP(raw.shape[1]); bug.add(raw)
    service.rank(queries[0]['query']) # warm up, excluded from timings
    records=[]
    for i,q in enumerate(queries):
        first,ranked,timing=service.rank(q['query'])
        vector=service.encoder.encode([preprocess(q['query'])],normalize=False)
        _,bad=bug.search(vector,10)
        records.append({**q,'bi':[r['case_id'] for r in first],'reranked':[r['case_id'] for r in ranked],
            'unnormalized':[service.rows[int(j)]['case_id'] for j in bad[0]],
            'top_score':ranked[0]['score'] if ranked else 0.,**timing})
        if (i+1)%25==0: print(f'Evaluated {i+1}/{len(queries)} queries',flush=True)
    answerable=[r for r in records if not r['no_answer']]; negatives=[r for r in records if r['no_answer']]
    thresholds=sorted({0.,1.,*[float(np.nextafter(r['top_score'],np.inf)) for r in records if r['top_score']<1]})
    def objective(t):
        return .5*(np.mean([r['top_score']<t for r in negatives])+np.mean([r['top_score']>=t for r in answerable]))
    threshold=max(thresholds,key=lambda t:(objective(t),-t))
    languages={r['case_id']:r['lang'] for r in service.rows}
    cross=[r for r in answerable if all(languages[k]!=r['lang'] for k in r['relevant_case_ids'])]
    same=[r for r in answerable if all(languages[k]==r['lang'] for k in r['relevant_case_ids'])]
    mixed=[r for r in answerable if r not in cross and r not in same]
    same_m=metrics(same,'reranked'); cross_m=metrics(cross,'reranked')
    report={'bi_encoder':metrics(answerable,'bi'),'reranked':metrics(answerable,'reranked'),
        'unnormalized':metrics(answerable,'unnormalized'),
        'mrr_lift':metrics(answerable,'reranked')['mrr_at_10']-metrics(answerable,'bi')['mrr_at_10'],
        'language_slices':{l:metrics([r for r in answerable if r['lang']==l],'reranked') for l in ['ar','en']},
        'cross_lingual':cross_m,'same_language':same_m,'mixed_language':metrics(mixed,'reranked'),
        'cross_lingual_mrr_gap':same_m['mrr_at_10']-cross_m['mrr_at_10'] if same_m and cross_m else None,
        'threshold':threshold,'no_answer_empty_correct':sum(r['top_score']<threshold for r in negatives),
        'no_answer_total':len(negatives),'answerable_retained':sum(r['top_score']>=threshold for r in answerable),
        'unique_no_answer_texts':len({r['query'] for r in negatives}),
        'threshold_scope':'Calibration on supplied 150 queries; no independent threshold test set',
        'latency_ms':{k:{'p50':float(np.median([r[k] for r in records])),'p99':float(np.percentile([r[k] for r in records],99))} for k in ['bi_ms','rerank_ms']},
        'raw_vector_norm_range':[float(np.linalg.norm(raw,axis=1).min()),float(np.linalg.norm(raw,axis=1).max())],
        'targets':{'recall':metrics(answerable,'reranked')['recall_at_10']>=.8,'mrr':metrics(answerable,'reranked')['mrr_at_10']>=.7,'empty':sum(r['top_score']<threshold for r in negatives)>=17},
        'limitations':'Synthetic corpus contains repeated texts with distinct case IDs. Exact-ID labels are sparse; semantically equivalent unlabelled cases still count as incorrect. No labels or corpus rows were changed.'}
    language_comparison={}
    for name,match in [('same_language',True),('cross_language',False)]:
        subset=[]
        for row in answerable:
            gold=[c for c in row['relevant_case_ids'] if (languages[c]==row['lang'])==match]
            if gold: subset.append({**row,'relevant_case_ids':gold})
        language_comparison[name]=metrics(subset,'reranked')
    language_comparison['mrr_gap_same_minus_cross']=language_comparison['same_language']['mrr_at_10']-language_comparison['cross_language']['mrr_at_10']
    language_comparison['definition']='Same queries and unfiltered top-10 ranking; separate relevant IDs by whether case and query languages match.'
    report['within_query_language_comparison']=language_comparison
    write_json(out/'query_results.json',records); write_json(out/'retrieval.json',report)
    write_json(out/'index_manifest.json',service.manifest)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
