"""Lab 2 contract checks."""
import torch
import torch.nn.functional as F
import pytest
from bayan.attention import MultiHeadAttention, attention


def test_attention_matches_pytorch():
    torch.manual_seed(42)
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    actual = attention(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_causal_mask_prevents_future_attention():
    torch.manual_seed(42)
    q, k, v = (torch.randn(2, 3, 5, 8) for _ in range(3))
    actual, weights = attention(q, k, v, is_causal=True, need_weights=True)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
    assert torch.count_nonzero(weights.triu(diagonal=1)) == 0
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 3, 5))
    changed_v = v.clone()
    changed_v[:, :, -1] += 100
    changed = attention(q, k, changed_v, is_causal=True)
    torch.testing.assert_close(changed[:, :, :-1], actual[:, :, :-1])


@pytest.mark.parametrize("additive", [False, True])
def test_padding_mask_matches_pytorch_and_prevents_leak(additive):
    torch.manual_seed(7)
    q, k, v = (torch.randn(2, 3, 6, 8) for _ in range(3))
    valid = torch.tensor([[True, True, True, False, False, False],
                          [True, True, True, True, False, False]])
    mask = valid[:, None, None, :]
    if additive:
        mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill(~mask, float("-inf"))
    actual, weights = attention(q, k, v, mask, need_weights=True)
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
    pad = ~valid[:, None, None, :]
    assert weights.masked_select(pad).sum() == 0
    _, unmasked = attention(q, k, v, need_weights=True)
    assert unmasked.masked_select(pad).sum() > 0
    changed_v = v.masked_fill(pad.transpose(-2, -1), 1000.0)
    torch.testing.assert_close(attention(q, k, changed_v, mask), actual)


def test_fully_masked_rows_have_zero_output_and_finite_gradients():
    torch.manual_seed(9)
    q, k, v = (torch.randn(1, 2, 3, 4, requires_grad=True) for _ in range(3))
    mask = torch.ones(3, 3, dtype=torch.bool)
    mask[1] = False
    output, weights = attention(q, k, v, mask, need_weights=True)
    torch.testing.assert_close(
        output, F.scaled_dot_product_attention(q, k, v, attn_mask=mask),
        atol=1e-6, rtol=0,
    )
    assert torch.count_nonzero(output[:, :, 1]) == 0
    assert torch.count_nonzero(weights[:, :, 1]) == 0
    output.sum().backward()
    assert all(torch.isfinite(tensor.grad).all() for tensor in (q, k, v))


@pytest.mark.parametrize("cross_attention", [False, True])
@pytest.mark.parametrize("additive", [False, True])
def test_mha_matches_pytorch_outputs_weights_and_gradients(cross_attention, additive):
    torch.manual_seed(12)
    actual_model = MultiHeadAttention(16, 4).double()
    reference = torch.nn.MultiheadAttention(16, 4, dropout=0, batch_first=True).double()
    with torch.no_grad():
        reference.in_proj_weight.copy_(torch.cat([
            actual_model.q_proj.weight, actual_model.k_proj.weight, actual_model.v_proj.weight
        ]))
        reference.in_proj_bias.copy_(torch.cat([
            actual_model.q_proj.bias, actual_model.k_proj.bias, actual_model.v_proj.bias
        ]))
        reference.out_proj.load_state_dict(actual_model.out_proj.state_dict())
    q = torch.randn(2, 4, 16, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 6 if cross_attention else 4, 16, dtype=torch.float64, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    pad = torch.zeros(k.shape[:2], dtype=torch.bool)
    pad[0, -2:] = True
    pad[1, -1] = True
    allowed = torch.ones(q.shape[1], k.shape[1], dtype=torch.bool).tril()
    mask = torch.zeros_like(allowed, dtype=q.dtype).masked_fill(~allowed, float("-inf")) if additive else allowed
    actual, weights = actual_model(q, k, v, mask, key_padding_mask=pad, need_weights=True)
    ref_mask = mask if additive else ~allowed  # nn.MHA uses True = blocked.
    ref_pad = torch.zeros_like(pad, dtype=q.dtype).masked_fill(pad, float("-inf")) if additive else pad
    expected, ref_weights = reference(
        q, k, v, attn_mask=ref_mask, key_padding_mask=ref_pad, average_attn_weights=False
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(weights, ref_weights, atol=1e-6, rtol=0)
    actual_grad = torch.autograd.grad(actual.square().sum(), (q, k, v))
    reference_grad = torch.autograd.grad(expected.square().sum(), (q, k, v))
    for actual_g, reference_g in zip(actual_grad, reference_grad):
        torch.testing.assert_close(actual_g, reference_g, atol=1e-6, rtol=0)


def test_mha_self_attention_defaults_and_parameter_registration():
    model = MultiHeadAttention(12, 3, bias=False)
    x = torch.randn(2, 5, 12)
    torch.testing.assert_close(model(x), model(x, x, x))
    assert sum(p.numel() for p in model.parameters()) == 4 * 12 * 12
    with pytest.raises(ValueError):
        MultiHeadAttention(10, 3)
