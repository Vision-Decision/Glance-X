"""Low-level layer kernels shared by every Glance-X direction.

A *prefix* is a sequence computed once and shared; a *block* is a sequence that continues a prefix:
  * Gated DeltaNet layers: a block starts from its prefix's final recurrent state and last (k-1) conv inputs, exactly
    what a causal continuation would see, but not the other blocks.
  * full-attention layers: a block attends to all valid prefix keys plus its own block (bidirectional inside the
    block with mask "block_bidir", causal with "causal"). The prefix itself is plain causal.
This equals running every (prefix + block) separately, with the prefix computed once.
"""
import torch
import torch.nn.functional as F

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gdr
except Exception:  # pragma: no cover
    fla_chunk_gdr = None


def _gdr(q, k, v, g, beta, initial_state=None):
    """Chunked gated delta rule; q,k:[B,T,Hk,Dk] v:[B,T,Hv,Dv] g,beta:[B,T,Hv]. Returns (o, final_state fp32)."""
    if fla_chunk_gdr is not None:
        return fla_chunk_gdr(q, k, v, g=g, beta=beta, initial_state=initial_state, output_final_state=True,
                             use_qk_l2norm_in_kernel=True)
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    return torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=initial_state,
                                        output_final_state=True, use_qk_l2norm_in_kernel=True)


def _gdn_mix(m, x):
    """Shared projections. x:[B,T,D] -> mixed_qkv [B,C,T], z, beta, g."""
    B, T, _ = x.shape
    mixed = m.in_proj_qkv(x).transpose(1, 2)
    z = m.in_proj_z(x).reshape(B, T, -1, m.head_v_dim)
    beta = m.in_proj_b(x).sigmoid()
    g = -m.A_log.float().exp() * F.softplus(m.in_proj_a(x).float() + m.dt_bias)
    return mixed, z, beta, g


def _gdn_core(m, conv_out, z, beta, g, init):
    B, _, T = conv_out.shape
    conv_out = F.silu(conv_out).transpose(1, 2)
    q, k, v = torch.split(conv_out, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
    q = q.reshape(B, T, -1, m.head_k_dim)
    k = k.reshape(B, T, -1, m.head_k_dim)
    v = v.reshape(B, T, -1, m.head_v_dim)
    o, S = _gdr(q, k, v, g, beta, init)
    o = m.norm(o.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim)).reshape(B, T, -1)
    return m.out_proj(o), S


def gdn_prefix(m, xp, p_valid):
    """Prefix half of a Gated DeltaNet layer. -> output, state (final recurrent state S, last k-1 conv inputs)."""
    K = m.conv_kernel_size
    w = m.conv1d.weight  # [C,1,K]
    xp = xp * p_valid[..., None].to(xp.dtype)               # left padded: zero the pads so they leave the state untouched
    mixed, z, beta, g = _gdn_mix(m, xp)
    Lp = mixed.shape[-1]
    conv = F.conv1d(mixed.to(w.dtype), w, m.conv1d.bias, padding=K - 1, groups=mixed.shape[1])[..., :Lp]
    op, S = _gdn_core(m, conv, z, beta, g, None)
    return op, (S, mixed[:, :, Lp - (K - 1):])


def gdn_block(m, xq, state, q_img=None):
    """Block half: every block continues from its prefix's state. q_img=None: block m belongs to prefix m."""
    S, tail = state
    if q_img is not None:
        S, tail = S[q_img], tail[q_img]
    w = m.conv1d.weight
    mixed_q, zq, beta_q, g_q = _gdn_mix(m, xq)
    mq = torch.cat([tail, mixed_q], dim=-1).to(w.dtype)
    conv_q = F.conv1d(mq, w, m.conv1d.bias, padding=0, groups=mq.shape[1])  # -> exactly Tq outputs
    oq, _ = _gdn_core(m, conv_q, zq, beta_q, g_q, S.contiguous())
    return oq


def gdn_forward(m, xp, xq, p_valid, q_img):
    op, state = gdn_prefix(m, xp, p_valid)
    return op, gdn_block(m, xq, state, q_img)


def _qkv(m, x, cos, sin):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    B, T, _ = x.shape
    q, gate = torch.chunk(m.q_proj(x).view(B, T, -1, m.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(B, T, -1)
    q = m.q_norm(q).transpose(1, 2)
    k = m.k_norm(m.k_proj(x).view(B, T, -1, m.head_dim)).transpose(1, 2)
    v = m.v_proj(x).view(B, T, -1, m.head_dim).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q, k, v, gate


def sdpa(q, k, v, mask, scale):
    """Masked attention with grouped KV heads. Repeating K/V to the query heads (instead of enable_gqa=True) lets
    PyTorch use the memory-efficient kernel with a boolean mask; enable_gqa + mask falls back to the slow math
    kernel. Same result."""
    g = q.shape[1] // k.shape[1]
    if g > 1:
        B, H, L, D = k.shape
        k = k[:, :, None].expand(B, H, g, L, D).reshape(B, H * g, L, D)
        v = v[:, :, None].expand(B, H, g, L, D).reshape(B, H * g, L, D)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)


def attn_prefix(m, xp, rope_p, mask_p):
    """Prefix half of an attention layer. -> output, state (the prefix keys and values)."""
    B, Lp, _ = xp.shape
    qp, kp, vp, gp = _qkv(m, xp, *rope_p)
    op = sdpa(qp, kp, vp, mask_p, m.scaling)
    return m.o_proj(op.transpose(1, 2).reshape(B, Lp, -1) * torch.sigmoid(gp)), (kp, vp)


def attn_block(m, xq, rope_q, mask_q, state, q_img=None):
    """Block half: each block attends to its prefix's keys + its own block. q_img=None: block m <-> prefix m."""
    kp, vp = state
    if q_img is not None:
        kp, vp = kp[q_img], vp[q_img]
    M, Tq, _ = xq.shape
    qq, kq, vq, gq = _qkv(m, xq, *rope_q)
    oq = sdpa(qq, torch.cat([kp, kq], dim=2), torch.cat([vp, vq], dim=2), mask_q, m.scaling)
    return m.o_proj(oq.transpose(1, 2).reshape(M, Tq, -1) * torch.sigmoid(gq))


def attn_forward(m, xp, xq, rope_p, rope_q, mask_p, mask_q, q_img):
    op, state = attn_prefix(m, xp, rope_p, mask_p)
    return op, attn_block(m, xq, rope_q, mask_q, state, q_img)


def prefix_mask(p_valid):
    """Causal mask over a left-padded prefix (pads attend to themselves, so no row is empty)."""
    Lp = p_valid.shape[1]
    causal = torch.ones(Lp, Lp, dtype=torch.bool, device=p_valid.device).tril()
    eye = torch.eye(Lp, dtype=torch.bool, device=p_valid.device)
    return ((causal[None] & p_valid[:, None, :]) | eye[None])[:, None]


def block_mask(p_valid_rows, q_valid, mode):
    """Mask of each block over [its prefix's keys | its own tokens]. p_valid_rows [M,Lp] (the prefix of each block)."""
    M, Tq = q_valid.shape
    Lp = p_valid_rows.shape[1]
    dev = q_valid.device
    if mode == "block_bidir":
        within = q_valid[:, None, :].expand(M, Tq, Tq)
    else:
        within = torch.ones(Tq, Tq, dtype=torch.bool, device=dev).tril()[None] & q_valid[:, None, :]
    within = within | torch.eye(Tq, dtype=torch.bool, device=dev)[None]
    return torch.cat([p_valid_rows[:, None, :].expand(M, Tq, Lp), within], dim=-1)[:, None]


def build_masks(p_valid, q_valid, q_img, mode):
    """Boolean SDPA masks (True = may attend)."""
    return prefix_mask(p_valid), block_mask(p_valid[q_img], q_valid, mode)
