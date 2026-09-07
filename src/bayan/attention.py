"""Lab 2: attention from scratch, with inspectable per-head probabilities."""
import math

import torch
from torch import nn


def attention(q, k, v, mask=None, *, is_causal=False, need_weights=False):
    """Compute softmax(Q K^T / sqrt(d_k) + mask) V, without dropout.

    Inputs end in (sequence, features); leading batch/head axes broadcast.
    A boolean mask uses True = allowed, like PyTorch's functional SDPA.
    Floating masks add to scores (0 = allowed, -inf = blocked). Masks must
    broadcast to (..., query_length, key_length). Fully blocked rows return
    zero weights/output. Optionally return (output, weights) for diagnostics.
    """
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise ValueError("Q/K feature sizes and K/V sequence lengths must match")
    if q.shape[-1] == 0:
        raise ValueError("Attention head dimension must be positive")
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        elif torch.is_floating_point(mask):
            scores = scores + mask
        else:
            raise TypeError("mask must be boolean or floating point")
    if is_causal:
        allowed = torch.ones(
            q.shape[-2], k.shape[-2], dtype=torch.bool, device=q.device
        ).tril()
        scores = scores.masked_fill(~allowed, float("-inf"))

    # Avoid softmax(-inf, ..., -inf), including NaN gradients on empty rows.
    blocked = torch.isneginf(scores).all(dim=-1, keepdim=True)
    weights = torch.softmax(scores.masked_fill(blocked, 0.0), dim=-1)
    weights = weights.masked_fill(blocked, 0.0)
    output = weights @ v
    return (output, weights) if need_weights else output


class MultiHeadAttention(nn.Module):
    """Batch-first MHA for (batch, sequence, embed_dim) inputs.

    ``mask`` has the same allowed-True convention as attention(). In contrast,
    ``key_padding_mask`` is (batch, key_length), True = padding to ignore,
    matching torch.nn.MultiheadAttention. Diagnostic weights retain heads.
    """

    def __init__(self, embed_dim, num_heads, bias=True):
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self, q, k=None, v=None, mask=None, *, key_padding_mask=None,
        is_causal=False, need_weights=False,
    ):
        k = q if k is None else k
        v = k if v is None else v
        for tensor in (q, k, v):
            if tensor.ndim != 3 or tensor.shape[-1] != self.embed_dim:
                raise ValueError("Inputs must have shape (batch, sequence, embed_dim)")
        if q.shape[0] != k.shape[0] or k.shape[:2] != v.shape[:2]:
            raise ValueError("Input batches and K/V sequence lengths must match")

        def split_heads(tensor):
            return tensor.reshape(
                tensor.shape[0], tensor.shape[1], self.num_heads, self.head_dim
            ).transpose(1, 2)

        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool or key_padding_mask.shape != k.shape[:2]:
                raise ValueError("key_padding_mask must be boolean with shape (batch, key_length)")
            allowed = ~key_padding_mask[:, None, None, :]
            if mask is None:
                mask = allowed
            elif mask.dtype == torch.bool:
                mask = mask & allowed
            elif torch.is_floating_point(mask):
                mask = mask.masked_fill(~allowed, float("-inf"))
            else:
                raise TypeError("mask must be boolean or floating point")

        output, weights = attention(
            split_heads(self.q_proj(q)), split_heads(self.k_proj(k)),
            split_heads(self.v_proj(v)), mask,
            is_causal=is_causal, need_weights=True,
        )
        output = output.transpose(1, 2).reshape(q.shape[0], q.shape[1], self.embed_dim)
        output = self.out_proj(output)
        return (output, weights) if need_weights else output
