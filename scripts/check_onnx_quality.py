"""Paired validation checks for fp32 and INT8; never rerun the frozen test."""
import argparse
import json
import numpy as np
import pandas as pd
from seqeval.metrics import f1_score as entity_f1
from seqeval.scheme import IOB2
from bayan.models.training import ROOT,write_json
from bayan.serving.runtime import Runtime
from bayan.evaluation.slices import f1_interval


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['classifier','ner'],default='classifier');parser.add_argument('--backend',choices=['onnx','int8'],required=True);args=parser.parse_args()
    path=ROOT/'artifacts/lab7'/f'{args.task}_{args.backend}_quality.json'
    if path.exists(): print(path.read_text());return
    runtime=Runtime(args.task,args.backend)
    original=json.loads((runtime.directory/'validation_predictions.json').read_text())
    if args.task=='classifier':
        frame=pd.DataFrame(original).rename(columns={'label':'y_true','prediction':'baseline'})
        raw=pd.read_csv(ROOT/'data/raw/bayan_feedback.csv');frame=frame.merge(raw[['feedback_id','text']],on='feedback_id',validate='one_to_one')
        frame['y_pred']=runtime.predict(frame.text.tolist())
        labels=list(runtime.labels.values())
        quality=f1_interval(frame,labels);tax=f1_interval(frame.assign(y_pred=frame.baseline),labels,other=frame.y_pred)
        agreement=float((frame.baseline==frame.y_pred).mean())
        predictions=frame[['feedback_id','y_true','y_pred']].to_dict('records')
    else:
        gold=[r['gold'] for r in original]; before=[r['prediction'] for r in original]; after=[]
        for offset in range(0,len(original),32):
            rows=original[offset:offset+32]
            batch=runtime.tokenize([r['tokens'] for r in rows],words=True)
            logits=runtime.logits(batch).argmax(-1)
            for i,row in enumerate(rows):
                ids=batch.word_ids(batch_index=i); chosen={}
                for j,word in enumerate(ids):
                    if word is not None and word not in chosen: chosen[word]=runtime.labels[int(logits[i,j])]
                if len(chosen)!=len(row['tokens']): raise ValueError('NER truncation in quality check')
                after.append([chosen[j] for j in range(len(row['tokens']))])
        def metric(g,p): return float(entity_f1(g,p,mode='strict',scheme=IOB2,zero_division=0))
        score=metric(gold,after); delta=metric(gold,before)-score
        manifest=json.loads((runtime.directory/'split_manifest.json').read_text())['validation']
        mapping={r['id']:r['template_group'] for r in manifest};groups={}
        for i,r in enumerate(original):groups.setdefault(mapping[r['id']],[]).append(i)
        arrays=list(groups.values());rng=np.random.default_rng(42);samples=[]
        for _ in range(500):
            ix=[i for group in rng.integers(0,len(arrays),len(arrays)) for i in arrays[group]]
            g=[gold[i] for i in ix];samples.append(metric(g,[before[i] for i in ix])-metric(g,[after[i] for i in ix]))
        quality={'point':score,'n':len(original)};tax={'point':delta,'low':float(np.quantile(samples,.025)),'high':float(np.quantile(samples,.975)),'groups':len(groups)}
        agreement=float(np.mean([a==b for a,b in zip(before,after)]));predictions=[{'id':r['id'],'gold':g,'prediction':p} for r,g,p in zip(original,gold,after)]
    report={'task':args.task,'backend':args.backend,'scope':'Paired validation; frozen test not used','quality':quality,'quality_tax_baseline_minus_candidate':tax,
        'agreement_with_saved_fp32':agreement,'quality_tax_within_1_point':tax['high']<=.01,'predictions_file':f'{args.task}_{args.backend}_predictions.json',
        'limitations':'Synthetic template data; bootstrap resamples citizen groups for classification and only four template groups for NER. No general-domain quality guarantee.'}
    write_json(path.parent/report['predictions_file'],predictions);write_json(path,report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
