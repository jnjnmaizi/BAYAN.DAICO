"""Bayan service assembling the measured lab artifacts."""
from contextlib import asynccontextmanager
from functools import lru_cache
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import json
import asyncio
import re
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from bayan.models.training import ROOT
from bayan.preprocessing.core import preprocess,mask_pii,PREPROC_VERSION
from bayan.serving.runtime import Runtime
from bayan.serving.canaries import run_startup_canaries
from bayan.serving.batching import MicroBatcher

class Request(BaseModel):
    text: str = Field(min_length=1,max_length=10000)

_init_lock=Lock()
_canaries=None
_batcher=None
_search_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='bayan-search')

@lru_cache(maxsize=1)
def selection(): return json.loads((ROOT/'artifacts/lab7/selection.json').read_text())

@lru_cache(maxsize=2)
def runtime(task): return Runtime(task,selection()[task]['backend'])

def ensure_ready():
    global _canaries
    if _canaries is None:
        with _init_lock:
            if _canaries is None: _canaries=run_startup_canaries(runtime('classifier'))
    return _canaries

@asynccontextmanager
async def lifespan(app):
    ensure_ready()
    yield

app=FastAPI(title='Bayan — Bilingual Citizen-Feedback Intelligence Service',lifespan=lifespan)

@app.get('/health')
def health():
    try: return {'status':'ready','canaries':ensure_ready(),'models':selection()}
    except (OSError,ValueError,AssertionError) as exc: raise HTTPException(503,str(exc))

@app.post('/v1/classify')
async def classify(payload:Request):
    global _batcher
    ensure_ready()
    model=runtime('classifier')
    loop=asyncio.get_running_loop()
    if _batcher is None or _batcher.loop is not loop:
        _batcher=MicroBatcher(model)
    try:
        logits=await _batcher.submit(payload.text)
    except asyncio.QueueFull:
        raise HTTPException(503,'Inference queue is full; retry later')
    probs=np.exp(logits-logits.max());probs/=probs.sum();pred=int(probs.argmax())
    return {'topic':model.labels[pred],'score':float(probs[pred]),'preproc_version':PREPROC_VERSION,'backend':model.backend}

@app.post('/v1/entities')
def entities(payload:Request):
    ensure_ready();model=runtime('ner');text=mask_pii(payload.text)
    tokens=list(re.finditer(r'\S+',text))
    if not tokens:return {'text':text,'entities':[]}
    batch=model.tokenize([[m.group() for m in tokens]],words=True)
    logits=model.logits(batch)[0];mapping={}
    for i,word in enumerate(batch.word_ids(0)):
        if word is not None and word not in mapping:mapping[word]=model.labels[int(logits[i].argmax())]
    spans=[]
    for i,token in enumerate(tokens):
        tag=mapping.get(i,'O')
        if tag=='O':continue
        prefix,label=tag.split('-',1)
        if prefix=='I' and spans and spans[-1]['label']==label and spans[-1]['last_word']==i-1:
            spans[-1]['end']=token.end();spans[-1]['last_word']=i
        else:spans.append({'label':label,'start':token.start(),'end':token.end(),'last_word':i})
    for span in spans:span['text']=text[span['start']:span['end']];span.pop('last_word')
    return {'text':text,'entities':spans,'offsets_refer_to':'returned PII-masked source text','truncated':len(mapping)<len(tokens)}

@lru_cache(maxsize=1)
def search_service():
    from bayan.search.service import CaseSearch
    return CaseSearch(str(ROOT/'artifacts/search/cases'))

@app.post('/v1/search')
async def search(payload:Request):
    ensure_ready()
    def infer():
        # Keep Torch/FAISS initialization and inference on one dedicated CPU
        # thread; moving the same encoder across worker pools crashes OpenMP
        # on the installed macOS stack.
        threshold=json.loads((ROOT/'artifacts/lab5/retrieval.json').read_text())['threshold']
        return {'results':search_service().search(payload.text,min_score=threshold),'threshold':threshold}
    return await asyncio.get_running_loop().run_in_executor(_search_executor,infer)

@app.post('/v1/analyse')
async def analyse(payload:Request):
    return {'classification':await classify(payload),'entities':await asyncio.to_thread(entities,payload),'search':await search(payload)}
