"""Check the Metal recurrent-state layout and branching against fp32 PyTorch."""

import numpy as np
import pytest
import torch


def test_delta_rule_branches_from_fp32_state():
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("requires an Apple GPU")

    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    from glancex._mac_kernels import delta_rule

    rng = torch.Generator().manual_seed(43)
    batch, length, heads, dim = 2, 21, 2, 32
    q, k, v = (torch.randn(batch, length, heads, dim, generator=rng) for _ in range(3))
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * dim**-0.5
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    g = -torch.rand(batch, length, heads, generator=rng)
    beta = torch.rand(batch, length, heads, generator=rng)

    prefix = 7
    to_mlx = lambda x: mx.array(x.numpy())
    p, st = delta_rule(*(to_mlx(x[:, :prefix]) for x in (q, k, v, g.exp(), beta)))
    rows = torch.tensor([1, 0])
    tail = (to_mlx(x[:, prefix:]) for x in (q, k, v, g.exp(), beta))
    y, final = delta_rule(*tail, st, to_mlx(rows.int()), save_state=True)

    ref, cached = torch_chunk_gated_delta_rule(
        q[:, :prefix] * dim**0.5, k[:, :prefix], v[:, :prefix], g[:, :prefix], beta[:, :prefix],
        output_final_state=True,
    )
    expected, expected_state = torch_chunk_gated_delta_rule(
        q[:, prefix:] * dim**0.5, k[:, prefix:], v[:, prefix:], g[:, prefix:], beta[:, prefix:],
        initial_state=cached[rows].contiguous(), output_final_state=True,
    )
    np.testing.assert_allclose(np.asarray(p), ref.numpy(), rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(np.asarray(y), expected.numpy(), rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(np.asarray(final), expected_state.transpose(-2, -1).numpy(), rtol=3e-5, atol=3e-6)
    assert final.dtype == mx.float32
