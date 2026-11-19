"""Glance-X backbone: one Qwen3.5 model, three directions. See docs/ARCHITECTURE.md.

Layers 0..K-1  (modality stage): the image stream and the text stream run separately (each causal), so both can be
                                 cached and reused.
Layers K..N-1  (fusion stage):   prefix / block computation with the roles set by the direction:
    image_first (i2t): prefix = image,                block = [text, R]          one image, many requests
    text_first  (t2i): prefix = text (without R),     block = [image, R]         one request, many images
    joint       (bi) : no prefix,                     block = [image, text, R]   everything sees everything
R is the <box_start> readout token, always the last valid token of a block. Positions are fixed for every direction:
image M-RoPE from 0, text from cfg.text_pos0 (independent of the image size), R after the text.

Batch layout (glancex.data.collate): image streams [B, Li] and text streams [M, Lt] are LEFT padded; request m is
asked against image q_img[m].
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ._layers import (_gdn_core, _gdn_mix, _qkv, attn_block, attn_forward, attn_prefix, block_mask, build_masks,
                      gdn_block, gdn_forward, gdn_prefix, prefix_mask, sdpa)

MODES = ("i2t", "t2i", "bi")
ALIASES = {"image_first": "i2t", "text_first": "t2i", "joint": "bi", "i2t": "i2t", "t2i": "t2i", "bi": "bi"}


def canonical_mode(mode):
    """'image_first' | 'text_first' | 'joint' (or i2t / t2i / bi) -> i2t / t2i / bi."""
    if mode not in ALIASES:
        raise ValueError(f"mode must be one of image_first, text_first, joint; got {mode!r}")
    return ALIASES[mode]


# ----------------------------------------------------------------------------------------- single-stream layers
def gdn_stream(m, x, valid):
    """Plain causal GDN over a LEFT-padded stream (pads zeroed so they leave the state at zero)."""
    K = m.conv_kernel_size
    w = m.conv1d.weight
    x = x * valid[..., None].to(x.dtype)
    mixed, z, beta, g = _gdn_mix(m, x)
    L = mixed.shape[-1]
    conv = F.conv1d(mixed.to(w.dtype), w, m.conv1d.bias, padding=K - 1, groups=mixed.shape[1])[..., :L]
    o, _ = _gdn_core(m, conv, z, beta, g, None)
    return o


def stream_mask(valid):
    L = valid.shape[1]
    causal = torch.ones(L, L, dtype=torch.bool, device=valid.device).tril()
    eye = torch.eye(L, dtype=torch.bool, device=valid.device)
    return ((causal[None] & valid[:, None, :]) | eye[None])[:, None]


def attn_stream(m, x, rope, mask):
    B, L, _ = x.shape
    q, k, v, gate = _qkv(m, x, *rope)
    o = sdpa(q, k, v, mask, m.scaling)
    return m.o_proj(o.transpose(1, 2).reshape(B, L, -1) * torch.sigmoid(gate))


# ----------------------------------------------------------------------------------------- block packing
def pack(segments, T):
    """Concatenate the valid tokens of several LEFT-padded segments per row into one RIGHT-padded block.
    segments: list of (x [M,L,D], valid [M,L], pos [3,M,L], flag [M,L] or None). Returns x [M,T,D], valid [M,T],
    pos [3,M,T] (pads continue the last position), the packed flags (one bool tensor per segment) and the index of
    each row's last valid token. Static shapes, no host sync (CUDA-graph safe); T must bound the valid tokens per row."""
    M, D = segments[0][0].shape[0], segments[0][0].shape[-1]
    dev = segments[0][0].device
    out = segments[0][0].new_zeros(M, T + 1, D)                              # column T collects the pads
    pos = torch.zeros(3, M, T + 1, dtype=torch.long, device=dev)
    flags = []
    off = torch.zeros(M, dtype=torch.long, device=dev)
    for x, v, p, f in segments:
        dest = torch.where(v, off[:, None] + v.long().cumsum(1) - 1, T)      # [M,L] destination of each token
        out.scatter_(1, dest[..., None].expand(-1, -1, D), x)
        pos.scatter_(2, dest[None].expand(3, -1, -1), p)
        fl = torch.zeros(M, T + 1, dtype=torch.bool, device=dev)
        if f is not None:
            fl.scatter_(1, dest, f & v)
        flags.append(fl[:, :T])
        off = off + v.long().sum(1)
    out, pos = out[:, :T].contiguous(), pos[:, :, :T]
    valid = torch.arange(T, device=dev)[None] < off[:, None]
    last = torch.gather(pos, 2, (off - 1).clamp(min=0)[None, :, None].expand(3, M, 1))
    step = (torch.arange(T, device=dev)[None] - off[:, None] + 1).clamp(min=0)
    pos = torch.where(valid[None], pos, last + step[None])
    return out, valid, pos, flags, off - 1


class GlanceXBackbone(torch.nn.Module):
    def __init__(self, hf_model, cfg):
        super().__init__()
        self.hf = hf_model
        self.cfg = cfg
        inner = hf_model.model
        self.visual = inner.visual
        self.lm = inner.language_model
        self.layer_types = list(hf_model.config.text_config.layer_types)
        self.hidden = hf_model.config.text_config.hidden_size
        self.K = cfg.bidir_k
        assert 0 <= self.K <= len(self.layer_types)

    @torch.no_grad()
    def image_features(self, pixel_values, grid_thw):
        out = self.hf.model.get_image_features(pixel_values, grid_thw, return_dict=True)
        return list(out.pooler_output)

    def rope(self, x, pos3):
        return self.lm.rotary_emb(x, pos3)

    def _ckpt(self, fn, *args):
        if self.cfg.grad_ckpt and self.training and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def unimodal(self, x, valid, pos, layers):
        rope = self.rope(x, pos)
        mask = stream_mask(valid)

        def fn(layer, ltype, x):
            h = layer.input_layernorm(x)
            a = gdn_stream(layer.linear_attn, h, valid) if ltype == "linear_attention" else attn_stream(layer.self_attn, h, rope, mask)
            x = x + a
            return x + layer.mlp(layer.post_attention_layernorm(x))

        for layer, ltype in layers:
            x = self._ckpt(fn, layer, ltype, x)
        return x

    def fusion(self, xp, p_valid, p_pos, xq, q_valid, q_pos, q_img, layers):
        rope_p, rope_q = self.rope(xp, p_pos), self.rope(xq, q_pos)
        mask_p, mask_q = build_masks(p_valid, q_valid, q_img, self.cfg.bidir_block_mask)

        def fn(layer, ltype, xp, xq):
            hp, hq = layer.input_layernorm(xp), layer.input_layernorm(xq)
            if ltype == "linear_attention":
                ap, aq = gdn_forward(layer.linear_attn, hp, hq, p_valid, q_img)
            else:
                ap, aq = attn_forward(layer.self_attn, hp, hq, rope_p, rope_q, mask_p, mask_q, q_img)
            xp, xq = xp + ap, xq + aq
            xp = xp + layer.mlp(layer.post_attention_layernorm(xp))
            xq = xq + layer.mlp(layer.post_attention_layernorm(xq))
            return xp, xq

        for layer, ltype in layers:
            xp, xq = self._ckpt(fn, layer, ltype, xp, xq)
        return xp, xq

    def forward(self, batch, img_feats, mode):
        """-> dict(r [M,D] readout, vis [V,N,D] image-token states, vis_index [M] row of vis for each question)."""
        mode = canonical_mode(mode)
        cfg = self.cfg
        emb = self.lm.embed_tokens
        xi = self._embed_images(batch, img_feats)                            # i-th image patch <- i-th feature row
        xt = emb(batch["txt_ids"])
        iv, tv = batch["img_valid"], batch["txt_valid"]
        ipos, tpos = batch["img_pos"], batch["txt_pos"]
        q_img = batch["q_img"]
        M = xt.shape[0]
        layers = list(zip(self.lm.layers, self.layer_types))
        uni, fus = layers[: self.K], layers[self.K:]
        if uni:
            xi = self.unimodal(xi, iv, ipos, uni)
            xt = self.unimodal(xt, tv, tpos, uni)
        img_flag = batch["img_mask"]
        if mode == "i2t":
            xq, qv, qpos, _, last = pack([(xt, tv, tpos, None)], max(tv.shape[1], cfg.pad_txt_block_to))
            hp, hq = self.fusion(xi, iv, ipos, xq, qv, qpos, q_img, fus)
            hp, hq = self.lm.norm(hp), self.lm.norm(hq)
            vis = self._gather_vis(hp, img_flag, batch["vis_valid"])
            vis_index = q_img
            q_valid = qv
        else:
            seg_img = (xi[q_img], iv[q_img], ipos[:, q_img], img_flag[q_img])
            if mode == "t2i":
                # prefix = text without R; block = image tokens + R (R = last column of the left-padded text stream)
                xp, pv, ppos = xt[:, :-1], tv[:, :-1], tpos[:, :, :-1]
                r_seg = (xt[:, -1:], tv[:, -1:], tpos[:, :, -1:], None)
                T = max(iv.shape[1] + 1, cfg.pad_t2i_block_to)          # static upper bounds (no host sync)
                xq, qv, qpos, flags, last = pack([seg_img, r_seg], T)
                prefix_of = torch.arange(M, device=xt.device)
            else:  # bi: no prefix (a few invalid pad slots keep the GDN conv tail well-defined)
                xp = xt.new_zeros(M, 4, xt.shape[-1])
                pv = torch.zeros(M, 4, dtype=torch.bool, device=xt.device)
                ppos = torch.zeros(3, M, 4, dtype=torch.long, device=xt.device)
                T = max(iv.shape[1] + tv.shape[1], cfg.pad_bi_block_to)
                xq, qv, qpos, flags, last = pack([seg_img, (xt, tv, tpos, None)], T)
                prefix_of = torch.arange(M, device=xt.device)
            hp, hq = self.fusion(xp, pv, ppos, xq, qv, qpos, prefix_of, fus)
            hq = self.lm.norm(hq)
            vis = self._gather_vis(hq, flags[0], batch["vis_valid"][q_img])
            vis_index = torch.arange(M, device=xt.device)
            q_valid = qv & ~flags[0]                                         # the block's non-image tokens
        r = hq[torch.arange(M, device=hq.device), last]
        # q_tok / q_valid / r_idx: the request's tokens (for the two-way decoder)
        return dict(r=r, vis=vis, vis_index=vis_index, q_tok=hq, q_valid=q_valid, r_idx=last)

    # ------------------------------------------------------------------ image cache (image_first)
    def _embed_images(self, batch, img_feats):
        xi = self.lm.embed_tokens(batch["img_ids"])
        feats = torch.cat(img_feats, 0) if isinstance(img_feats, (list, tuple)) else img_feats
        im = batch["img_mask"]
        idx = (im.flatten().long().cumsum(0) - 1).clamp(min=0).view_as(im)
        return torch.where(im[..., None], feats[idx].to(xi.dtype), xi)

    @torch.no_grad()
    def encode_images(self, batch, img_feats):
        """image_first prefix only (it never depends on a request): every layer's state for every image.
        -> dict(states [per fusion layer: GDN (S, conv tail) | attention (K, V)], p_valid, vis [B,N,D])."""
        xi = self._embed_images(batch, img_feats)
        iv, ipos = batch["img_valid"], batch["img_pos"]
        layers = list(zip(self.lm.layers, self.layer_types))
        if self.K:
            xi = self.unimodal(xi, iv, ipos, layers[: self.K])
        rope_p, mask_p = self.rope(xi, ipos), prefix_mask(iv)
        states = []
        for layer, ltype in layers[self.K:]:
            h = layer.input_layernorm(xi)
            if ltype == "linear_attention":
                o, st = gdn_prefix(layer.linear_attn, h, iv)
            else:
                o, st = attn_prefix(layer.self_attn, h, rope_p, mask_p)
            xi = xi + o
            xi = xi + layer.mlp(layer.post_attention_layernorm(xi))
            states.append(st)
        vis = self._gather_vis(self.lm.norm(xi), batch["img_mask"], batch["vis_valid"])
        return dict(states=states, p_valid=iv, vis=vis)

    @torch.no_grad()
    def query_bank(self, bank, batch, q_img=None):
        """Request blocks against cached image states (image_first). batch: txt_ids / txt_valid / txt_pos [M,Lt].
        q_img [M]: image of each request; None = request m asks image m. A single request row is computed once
        and shared by every image."""
        cfg = self.cfg
        xt, tv, tpos = self.lm.embed_tokens(batch["txt_ids"]), batch["txt_valid"], batch["txt_pos"]
        layers = list(zip(self.lm.layers, self.layer_types))
        if self.K:
            xt = self.unimodal(xt, tv, tpos, layers[: self.K])
        n_img = bank["p_valid"].shape[0]
        if q_img is None and xt.shape[0] == 1 and n_img > 1:                   # one request, every image
            xt, tv, tpos = xt.expand(n_img, -1, -1), tv.expand(n_img, -1), tpos.expand(-1, n_img, -1)
        M = xt.shape[0]
        xq, qv, qpos, _, last = pack([(xt, tv, tpos, None)], max(tv.shape[1], cfg.pad_txt_block_to))
        p_rows = bank["p_valid"] if q_img is None else bank["p_valid"][q_img]
        rope_q, mask_q = self.rope(xq, qpos), block_mask(p_rows, qv, cfg.bidir_block_mask)
        for (layer, ltype), st in zip(layers[self.K:], bank["states"]):
            h = layer.input_layernorm(xq)
            if ltype == "linear_attention":
                o = gdn_block(layer.linear_attn, h, st, q_img)
            else:
                o = attn_block(layer.self_attn, h, rope_q, mask_q, st, q_img)
            xq = xq + o
            xq = xq + layer.mlp(layer.post_attention_layernorm(xq))
        hq = self.lm.norm(xq)
        ar = torch.arange(M, device=hq.device)
        return dict(r=hq[ar, last], vis=bank["vis"], vis_index=ar if q_img is None else q_img, q_tok=hq, q_valid=qv,
                    r_idx=last)

    @staticmethod
    def _gather_vis(h, flag, vis_valid):
        """Row r's flagged tokens, in order, into vis[r, :n_r] (static shapes, no host sync)."""
        R, N = vis_valid.shape
        j = torch.where(flag, flag.long().cumsum(1) - 1, N)                 # rank among flagged tokens; others -> N
        vis = h.new_zeros(R, N + 1, h.shape[-1])
        vis.scatter_(1, j[..., None].expand(-1, -1, h.shape[-1]), h)
        return vis[:, :N]
