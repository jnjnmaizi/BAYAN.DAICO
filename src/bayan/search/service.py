"""Bi-encoder retrieval followed by multilingual cross-encoder reranking."""
import json
import time
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bayan.search.index import Encoder, sources, ENCODER, RERANKER
from bayan.models.training import sha256
from bayan.preprocessing.core import preprocess, PREPROC_VERSION


class CaseSearch:
    def __init__(self, prefix: str, device='cpu'):
        self.manifest = json.load(open(f'{prefix}_manifest.json'))
        src = sources()
        m = self.manifest
        if m['preproc_version'] != PREPROC_VERSION or not m.get('normalized'):
            raise ValueError('Preprocessing/normalization manifest mismatch')
        for key, path in [('index_sha256',f'{prefix}.faiss'),('metadata_sha256',f'{prefix}_metadata.json')]:
            if sha256(path) != m[key]:
                raise ValueError('Index artifact checksum mismatch')
        if m['model_revision'] != src[ENCODER]['revision'] or m['reranker_revision'] != src[RERANKER]['revision']:
            raise ValueError('Model revision mismatch')
        self.index = faiss.read_index(f'{prefix}.faiss')
        self.rows = json.load(open(f'{prefix}_metadata.json'))
        if self.index.ntotal != len(self.rows) or self.index.ntotal != m['n_vectors'] or self.index.d != m['dim']:
            raise ValueError('Index dimensions or metadata count mismatch')
        self.encoder = Encoder(src[ENCODER]['path'], device)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(src[RERANKER]['path'],local_files_only=True)
        self.reranker = AutoModelForSequenceClassification.from_pretrained(src[RERANKER]['path'],local_files_only=True).to(device).eval()

    def rank(self, query, candidates=50):
        if not query.strip():
            return [], [], {'bi_ms':0.,'rerank_ms':0.}
        start=time.perf_counter()
        query = preprocess(query)
        vector = self.encoder.encode([query])
        scores, ids = self.index.search(vector, min(candidates,self.index.ntotal))
        first = [{**self.rows[int(i)],'bi_score':float(s)} for s,i in zip(scores[0],ids[0]) if i >= 0]
        bi_ms=(time.perf_counter()-start)*1000
        start=time.perf_counter()
        values=[]
        with torch.inference_mode():
            for offset in range(0,len(first),32):
                rows=first[offset:offset+32]
                batch=self.tokenizer([query]*len(rows),[preprocess(r['case_text']) for r in rows],padding=True,truncation=True,max_length=256,return_tensors='pt').to(self.device)
                values.extend(torch.sigmoid(self.reranker(**batch).logits.flatten()).cpu().tolist())
        ranked=sorted([{**r,'score':float(s)} for r,s in zip(first,values)],key=lambda r:(-r['score'],r['case_id']))
        return first,ranked,{'bi_ms':bi_ms,'rerank_ms':(time.perf_counter()-start)*1000}

    def search(self, query: str, k: int = 5, candidates: int = 50, min_score: float = 0.25):
        if k < 1 or candidates < k or not 0 <= min_score <= 1:
            raise ValueError('Require candidates >= k > 0 and threshold in [0,1]')
        _,ranked,_=self.rank(query,candidates)
        return [r for r in ranked if r['score'] >= min_score][:k]
