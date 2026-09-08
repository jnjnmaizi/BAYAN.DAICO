"""Constrained extractive QA decoding; offsets always refer to original context."""
import math

import numpy as np


def best_span(start_logits, end_logits, offsets, *, null_score, null_threshold, max_answer_len=30, top_k=20):
    """Return character offsets in ``answer``, or None if no answer is supported.

    Context-only tokens have nonempty (start, end) offsets; question/special/pad
    tokens must have None. Null wins when null_score - best_score > threshold,
    matching SQuAD2 score-difference decoding. max_answer_len is in tokens.
    """
    start, end = np.asarray(start_logits), np.asarray(end_logits)
    if start.ndim != 1 or end.shape != start.shape or len(offsets) != len(start):
        raise ValueError("Logits and offsets must be aligned one-dimensional sequences")
    if max_answer_len < 1 or top_k < 1:
        raise ValueError("max_answer_len and top_k must be positive")
    if not math.isfinite(null_score) or not math.isfinite(null_threshold):
        raise ValueError("Null score and threshold must be finite")
    valid = [i for i, offset in enumerate(offsets)
             if offset is not None and 0 <= offset[0] < offset[1]]
    starts = sorted((i for i in valid if np.isfinite(start[i])), key=lambda i: (-start[i], i))[:top_k]
    ends = sorted((i for i in valid if np.isfinite(end[i])), key=lambda i: (-end[i], i))[:top_k]
    valid_set, best = set(valid), None
    for i in starts:
        for j in ends:
            if j < i or j - i + 1 > max_answer_len:
                continue
            if any(k not in valid_set for k in range(i, j + 1)):
                continue
            if any(offsets[k][0] < offsets[k-1][0] or offsets[k][1] < offsets[k-1][1] for k in range(i+1, j+1)):
                continue
            score = float(start[i] + end[j])
            if best is None or score > best["score"]:
                best = {"answer": (offsets[i][0], offsets[j][1]), "score": score,
                        "start_token": i, "end_token": j}
    if best is None:
        return {"answer": None, "score": None, "score_diff": None}
    best["score_diff"] = float(null_score - best["score"])
    if best["score_diff"] > null_threshold:
        best["answer"] = None
    return best
