"""Versioned cosine-similarity FAISS index for the supplied case corpus."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import faiss
from transformers import AutoModel, AutoTokenizer
from bayan.models.training import ROOT, sha256, write_json
from bayan.preprocessing.core import preprocess, PREPROC_VERSION

ENCODER = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
RERANKER = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'


def sources():
    result = json.loads((ROOT / 'artifacts/lab5/model_sources.json').read_text())
    for source in result.values():
        source['path'] = str(ROOT / source['path'])
    return result


class Encoder:
    def __init__(self, source, device='cpu'):
        self.device = device
        # The installed macOS Torch/FAISS OpenMP combination crashes in
        # multi-threaded CPU embedding; one CPU thread is verified stable.
        torch.set_num_threads(1 if device == 'cpu' else 4)
        faiss.omp_set_num_threads(1)
        self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        self.model = AutoModel.from_pretrained(source, local_files_only=True).eval().to(device)

    def encode(self, texts, batch_size=64, normalize=True):
        output = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = self.tokenizer(texts[start:start+batch_size], padding=True, truncation=True, max_length=128, return_tensors='pt').to(self.device)
                hidden = self.model(**batch).last_hidden_state
                mask = batch['attention_mask'].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                output.append(pooled.cpu().numpy())
        result = np.ascontiguousarray(np.concatenate(output), dtype=np.float32)
        if normalize:
            faiss.normalize_L2(result)
        return result


def build_index(prefix, limit=None, device='cpu'):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    src = sources()
    corpus = ROOT / 'data/search/bayan_cases.csv'
    rows = pd.read_csv(corpus).head(limit).to_dict('records') if limit else pd.read_csv(corpus).to_dict('records')
    texts = [preprocess(r['case_text']) for r in rows]
    # Deduplicate computation, never case IDs or the corpus: repeated cases remain searchable.
    unique = list(dict.fromkeys(texts))
    encoder = Encoder(src[ENCODER]['path'], device)
    raw = encoder.encode(unique, normalize=False)
    lookup = dict(zip(unique, raw))
    vectors = np.ascontiguousarray([lookup[t] for t in texts], dtype=np.float32)
    np.save(f'{prefix}_raw.npy', vectors)
    faiss.normalize_L2(vectors)
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError('Index vectors must be finite unit vectors')
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, f'{prefix}.faiss')
    write_json(f'{prefix}_metadata.json', rows)
    manifest = {'model':ENCODER,'model_revision':src[ENCODER]['revision'],
        'reranker':RERANKER,'reranker_revision':src[RERANKER]['revision'],
        'preproc_version':PREPROC_VERSION,'n_vectors':len(rows),'dim':vectors.shape[1],
        'metric':'cosine via normalized inner product','normalized':True,'max_length':128,
        'data_sha256':sha256(corpus),'index_sha256':sha256(f'{prefix}.faiss'),
        'metadata_sha256':sha256(f'{prefix}_metadata.json')}
    write_json(f'{prefix}_manifest.json', manifest)
    return manifest
