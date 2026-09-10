"""Experimental hybrid retrieval over distinct text groups.

No relevance labels, case IDs or topic labels are ranking features. RRF scores
are ordering scores, not probabilities or calibrated no-answer thresholds.
"""
from collections import defaultdict

import faiss
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from bayan.preprocessing.core import preprocess


def reciprocal_rank_fusion(rankings, constant=60):
    if constant < 1:
        raise ValueError('RRF constant must be positive')
    scores = defaultdict(float)
    order = {}
    for ranking in rankings:
        if len(ranking) != len(set(ranking)):
            raise ValueError('Each ranking must contain unique candidates')
        for rank, item in enumerate(ranking, 1):
            order.setdefault(item, len(order))
            scores[item] += 1 / (constant + rank)
    return sorted(scores, key=lambda item: (-scores[item], order[item])), dict(scores)


class HybridGroupSearch:
    """Reuse a validated CaseSearch instance; preserve its default behavior."""
    def __init__(self, service):
        self.service = service
        self.texts = []
        self.members = []
        lookup = {}
        representatives = []
        for i, case in enumerate(service.rows):
            text = preprocess(case['case_text'])
            if text not in lookup:
                lookup[text] = len(self.texts)
                self.texts.append(text)
                self.members.append([])
                representatives.append(i)
            self.members[lookup[text]].append(dict(case))
        vectors = service.index.reconstruct_n(0, service.index.ntotal)[representatives]
        self.index = faiss.IndexFlatIP(service.index.d)
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self.lexical = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)
        self.matrix = self.lexical.fit_transform(self.texts)

    def rank(self, query, candidates=50, k=10):
        if k < 1 or candidates < k:
            raise ValueError('Require candidates >= k > 0')
        if not query.strip():
            return {'dense': [], 'lexical': [], 'cross_encoder': [], 'hybrid': [],
                    'original_grouped_cross_encoder': [], 'candidate_groups': 0}
        query = preprocess(query)
        vector = self.service.encoder.encode([query])
        _, indices = self.index.search(vector, min(candidates, self.index.ntotal))
        dense = [int(i) for i in indices[0] if i >= 0]
        similarity = (self.matrix @ self.lexical.transform([query]).T).toarray().ravel()
        lexical = [int(i) for i in np.argsort(-similarity, kind='stable')[:candidates] if similarity[i] > 0]
        union = list(dict.fromkeys(dense + lexical))
        scores = {}
        with torch.inference_mode():
            for start in range(0, len(union), 32):
                ids = union[start:start+32]
                inputs = self.service.tokenizer([query]*len(ids), [self.texts[i] for i in ids],
                                                truncation=True, padding=True, max_length=256,
                                                return_tensors='pt').to(self.service.device)
                values = torch.sigmoid(self.service.reranker(**inputs).logits.flatten()).cpu().tolist()
                scores.update(zip(ids, values))
        cross = sorted(union, key=lambda i: -scores[i])
        original = sorted(dense, key=lambda i: -scores[i])
        hybrid, fused = reciprocal_rank_fusion([dense, lexical, cross], constant=60)

        def pack(ids):
            return [{'case_text': self.texts[i], 'cases': self.members[i],
                     'cross_encoder_score': scores[i], 'rrf_score': fused[i]} for i in ids[:k]]
        return {'dense': pack(dense), 'lexical': pack(lexical), 'cross_encoder': pack(cross),
                'original_grouped_cross_encoder': pack(original), 'hybrid': pack(hybrid),
                'candidate_groups': len(union)}
