"""CPU-only inference ladder; seeded sample from the supplied production mix."""
import argparse
import json
import time
import platform
from pathlib import Path
import numpy as np
from bayan.models.training import ROOT,write_json,sha256
from bayan.serving.runtime import Runtime


def benchmark(runtime,texts,*,max_length=128,padding=True,warmup=10):
    # Tokenization excluded from bare model timings and separately included by HTTP load test.
    batches=[runtime.tokenize([t],max_length=max_length,padding=padding) for t in texts]
    for batch in batches[:warmup]: runtime.logits(batch)
    times=[]
    for batch in batches:
        start=time.perf_counter(); result=runtime.logits(batch); times.append((time.perf_counter()-start)*1000)
        if not np.isfinite(result).all(): raise ValueError('Nonfinite logits')
    return {'p50_ms':float(np.median(times)),'p99_ms':float(np.percentile(times,99)),
        'n':len(times),'warmup':min(warmup,len(batches)),'raw_ms':times,
        'token_lengths':{str(k):sum(int(b['input_ids'].shape[1])==k for b in batches) for k in sorted({int(b['input_ids'].shape[1]) for b in batches})}}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--backend',choices=['torch','onnx','int8'],default='torch'); parser.add_argument('--task',choices=['classifier','ner'],default='classifier');parser.add_argument('--samples',type=int,default=200); args=parser.parse_args()
    dest=ROOT/'artifacts/lab7'/f'{args.task}_{args.backend}_benchmark.json'
    if dest.exists(): print(dest.read_text()); return
    mix=ROOT/'data/serving/bench_mix.npy'; texts=np.load(mix)
    indices=np.random.default_rng(42).choice(len(texts),min(args.samples,len(texts)),replace=False)
    runtime=Runtime(args.task,args.backend,threads=4)
    rows={}
    if args.backend=='torch': rows['padded_512']=benchmark(runtime,texts[indices],max_length=512,padding='max_length')
    rows['dynamic_128']=benchmark(runtime,texts[indices],max_length=128,padding=True)
    if args.backend=='torch': size=(runtime.directory/'model.safetensors').stat().st_size
    else: size=sum(p.stat().st_size for p in runtime.path.parent.glob(runtime.path.name+'*'))
    write_json(dest,{'task':args.task,'backend':args.backend,'threads':4,'platform':platform.platform(),
        'mix_sha256':sha256(mix),'sample_indices':indices.tolist(),'sampling':'200 seeded rows without replacement from supplied 2000-row mix; no timing-result selection',
        'artifact_bytes':size,'rows':rows})
    print(args.task,args.backend,{k:{m:v[m] for m in ['p50_ms','p99_ms','n']} for k,v in rows.items()},flush=True)


if __name__=='__main__': main()
