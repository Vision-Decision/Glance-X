"""Qwen3.5 layers for MLX inference, loaded from the existing Torch model."""

import math
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
import torch
from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids

from ._mac_kernels import delta_rule, depthwise_silu


def array(tensor, dtype=mx.float32):
    dtype_torch = torch.float16 if dtype == mx.float16 else torch.float32
    return mx.array(tensor.detach().to(device="cpu", dtype=dtype_torch).contiguous().numpy())


class Linear:
    def __init__(self, *layers, dtype=mx.float16, weight=None, contiguous=False):
        if weight is None:
            weight = layers[0].weight if len(layers) == 1 else torch.cat([layer.weight for layer in layers])
        self.w = array(weight, dtype).T
        self.compute_dtype = dtype
        if contiguous:
            self.w = mx.contiguous(self.w)
            mx.eval(self.w)
        biases = [layer.bias for layer in layers]
        self.bias = None if all(b is None for b in biases) else array(
            torch.cat([b if b is not None else layer.weight.new_zeros(layer.weight.shape[0])
                       for b, layer in zip(biases, layers)]),
            dtype,
        )

    def __call__(self, x, compute_dtype=None):
        dtype = self.compute_dtype if compute_dtype is None else compute_dtype
        flat = x.astype(dtype).reshape(-1, x.shape[-1])
        weight = self.w.astype(dtype)
        if self.bias is not None:
            y = mx.addmm(self.bias.astype(dtype), flat, weight)
        elif dtype == mx.float16 and flat.shape[0] <= 16:
            # Metal's narrow-output kernel streams these weights faster with M and N exchanged.
            y = (weight.T @ flat.T).T
        else:
            y = flat @ weight
        return y.reshape(*x.shape[:-1], self.w.shape[-1])


def layer_norm(mod):
    weight, bias, eps = array(mod.weight), array(mod.bias), mod.eps
    return lambda x: mx.fast.layer_norm(x, weight, bias, eps)


def rms_norm(mod):
    weight, eps = array(1 + mod.weight.detach().float()), mod.eps
    return lambda x: mx.fast.rms_norm(x.astype(mx.float32), weight, eps)


@partial(mx.compile, shapeless=True)
def _swi_glu(gate, up):
    f = gate.astype(mx.float32)
    return (f * mx.sigmoid(f) * up).astype(gate.dtype)


@mx.compile
def _rotate(x, cos, sin):
    dim = cos.shape[-1]
    prefix = x[..., :dim].astype(mx.float32)
    a, b = mx.split(prefix, 2, axis=-1)
    rotated = mx.concatenate([-b, a], axis=-1)
    y = (prefix * cos + rotated * sin).astype(x.dtype)
    return mx.concatenate([y, x[..., dim:]], axis=-1) if dim < x.shape[-1] else y


@partial(mx.compile, shapeless=True)
def _l2(x, scale):
    x = x.astype(mx.float32)
    return x * (scale * mx.rsqrt(mx.sum(x * x, axis=-1, keepdims=True) + 1e-6))


@partial(mx.compile, shapeless=True)
def _decay(a, log_scale, bias):
    return mx.exp(-mx.exp(log_scale) * nn.softplus(a.astype(mx.float32) + bias))


@partial(mx.compile, shapeless=True)
def _gated_norm(x, z, weight):
    z = z.astype(mx.float32)
    return mx.fast.rms_norm(x, weight, 1e-6) * z * mx.sigmoid(z)


class PatchProjection:
    def __init__(self, proj):
        # The preprocessing repeats the still image in the two temporal planes.
        # Sum the stored fp16 planes in fp32 before applying the projection.
        self.planes = array(proj.weight.detach().transpose(1, 2).flatten(2), mx.float16)
        self.bias = array(proj.bias)

    def __call__(self, patches):
        weight = self.planes.astype(mx.float32).sum(axis=1).T
        return mx.addmm(self.bias, patches, weight)


class Vision:
    def __init__(self, vision, processor, precise=True):
        self.patch = PatchProjection(vision.patch_embed.proj)
        self.side = vision.num_grid_per_side
        self.merge = vision.spatial_merge_size
        self.patch_size = vision.patch_size
        self.pos_embed = array(vision.pos_embed.weight)
        self.inv_freq = array(vision.rotary_pos_emb.inv_freq)
        self.blocks = [
            (layer_norm(b.norm1), Linear(b.attn.qkv, contiguous=True), Linear(b.attn.proj), b.attn.num_heads,
             layer_norm(b.norm2), Linear(b.mlp.linear_fc1), Linear(b.mlp.linear_fc2))
            for b in vision.blocks
        ]
        self.m_norm = layer_norm(vision.merger.norm)
        self.m_up, self.m_down = Linear(vision.merger.linear_fc1), Linear(vision.merger.linear_fc2)
        self.rescale = processor.rescale_factor
        self.mean, self.std = mx.array(processor.image_mean), mx.array(processor.image_std)
        self.positions = {}
        self.forward = mx.compile(self._forward)
        if precise:
            self.use_fp32_arithmetic()

    def use_fp32_arithmetic(self):
        for block in self.blocks:
            for projection in (block[1], block[2], *block[5:]):
                projection.compute_dtype = mx.float32
        self.m_up.compute_dtype = self.m_down.compute_dtype = mx.float32

    def geometry(self, height, width):
        key = (height, width)
        if key not in self.positions:
            grid = torch.tensor([[1, height, width]])
            indices, weights = get_vision_interpolation_indices_and_weights(
                grid, self.side, align_corners=True, spatial_merge_size=self.merge,
            )
            coords = get_vision_position_ids(grid, self.merge)
            freq = (mx.array(coords.numpy().astype("int32"))[..., None] * self.inv_freq).reshape(height * width, -1)
            freq = mx.concatenate([freq, freq], axis=-1)[None, :, None]
            pos = (self.pos_embed[mx.array(indices.numpy().astype("int32"))] * array(weights)[..., None]).sum(axis=1)
            values = pos, mx.cos(freq), mx.sin(freq)
            mx.eval(values)
            self.positions[key] = values
        return self.positions[key]

    def __call__(self, image):
        height, width = (s // self.patch_size for s in image.shape[:2])
        return self.forward(image, *self.geometry(height, width))

    def _forward(self, image, pos, cos, sin):
        height, width = (s // self.patch_size for s in image.shape[:2])
        m, p = self.merge, self.patch_size
        pixels = (image.astype(mx.float32) * self.rescale - self.mean) / self.std
        patches = pixels.reshape(height // m, m, p, width // m, m, p, 3).transpose(0, 3, 1, 4, 6, 2, 5)
        h = self.patch(patches.reshape(height * width, -1)).astype(mx.float32) + pos
        for norm1, qkv, proj, heads, norm2, up, down in self.blocks:
            q, k, v = mx.split(qkv(norm1(h)).reshape(1, -1, 3, heads, h.shape[-1] // heads), 3, axis=2)
            q, k, v = (x.squeeze(2) for x in (q, k, v))
            q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
            a = mx.fast.scaled_dot_product_attention(
                q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3),
                scale=q.shape[-1] ** -0.5, force_fused=True,
            )
            h = h + proj(a.transpose(0, 2, 1, 3).reshape(h.shape))
            h = h + down(nn.gelu_approx(up(norm2(h))))
        merged = self.m_norm(h).reshape(h.shape[0] // (m * m), -1)
        return self.m_down(nn.gelu(self.m_up(merged))).astype(mx.float32)


class MPSVision:
    """The frozen HF vision tower with fp16 stored weights and fp32 arithmetic on MPS."""

    def __init__(self, vision, processor):
        self.module = vision.eval()
        for p in vision.parameters():
            p.data = p.data.to(dtype=torch.float16)
        self.module = self.module.to("mps")
        self.processor = processor
        self.patch_size = vision.patch_size
        self.merge = vision.spatial_merge_size

    @torch.inference_mode()
    def __call__(self, pixels):
        img = Image.fromarray(np.asarray(pixels))
        proc = self.processor(images=[img], do_resize=False, return_tensors="pt").to("mps")
        weights = {name: p.float() for name, p in self.module.named_parameters()}
        feat = torch.func.functional_call(
            self.module, weights, (proc["pixel_values"], proc["image_grid_thw"]),
        ).pooler_output
        return array(feat)


class GatedDelta:
    def __init__(self, mod):
        self.in_proj = Linear(mod.in_proj_qkv, mod.in_proj_z, mod.in_proj_b, mod.in_proj_a, contiguous=True)
        self.out_proj = Linear(mod.out_proj)
        self.splits = [2 * mod.key_dim + mod.value_dim, 2 * mod.key_dim + 2 * mod.value_dim]
        self.splits.append(self.splits[-1] + mod.num_v_heads)
        self.log_scale, self.bias = array(mod.A_log), array(mod.dt_bias)
        self.conv = array(mod.conv1d.weight.transpose(1, 2), mx.float16)
        self.norm = array(mod.norm.weight)
        self.hk, self.hv = mod.num_k_heads, mod.num_v_heads
        self.dk, self.dv = mod.head_k_dim, mod.head_v_dim

    def project(self, x, compute_dtype=None):
        return mx.split(self.in_proj(x, compute_dtype), self.splits, axis=-1)

    def mix(self, projected, state=None, rows=None, save_state=False):
        qkv, z, b, a = projected
        batch, length, channels = qkv.shape
        keep = self.conv.shape[1] - 1
        if state is None:
            initial, tail = None, mx.zeros((batch, keep, channels), qkv.dtype)
        else:
            initial, tail = state
            if rows is not None:
                tail = tail[rows]
        history = mx.concatenate([tail, qkv], axis=1)
        mixed = depthwise_silu(history, self.conv)
        q, k, v = mx.split(mixed, [self.hk * self.dk, 2 * self.hk * self.dk], axis=-1)
        q = _l2(q.reshape(batch, length, self.hk, self.dk), mx.array(self.dk**-0.5))
        k = _l2(k.reshape(batch, length, self.hk, self.dk), mx.array(1.0))
        v = v.reshape(batch, length, self.hv, self.dv)
        vectorize = length >= 128 or batch >= 64
        o, final = delta_rule(
            q, k, v, _decay(a, self.log_scale, self.bias), mx.sigmoid(b.astype(mx.float32)), initial, rows, save_state,
            width=8 if vectorize else 16, vector=4 if vectorize else 1,
        )
        y = _gated_norm(o, z.reshape(v.shape), self.norm).reshape(batch, length, -1)
        return y, (final, history[:, -keep:]) if save_state else None


class Attention:
    def __init__(self, mod):
        self.in_proj = Linear(mod.q_proj, mod.k_proj, mod.v_proj)
        self.out_proj = Linear(mod.o_proj)
        self.q_norm, self.k_norm = rms_norm(mod.q_norm), rms_norm(mod.k_norm)
        self.scale, self.d = mod.scaling, mod.head_dim
        self.splits = [mod.q_proj.out_features, mod.q_proj.out_features + mod.k_proj.out_features]

    def project(self, x, compute_dtype=None):
        return mx.split(self.in_proj(x, compute_dtype), self.splits, axis=-1)

    def project_readout(self, x):
        # Only the last query is consumed at the end of a cached search.
        q_width = self.splits[0]
        flat = x.astype(self.in_proj.w.dtype).reshape(-1, x.shape[-1])
        last = x[:, -1:].astype(flat.dtype).reshape(-1, flat.shape[-1])
        q = (last @ self.in_proj.w[:, :q_width]).reshape(x.shape[0], 1, -1)
        kv = (flat @ self.in_proj.w[:, q_width:]).reshape(*x.shape[:2], -1)
        return (q, *mx.split(kv, [self.splits[1] - q_width], axis=-1))

    def mix(self, projected, rope, mask, state=None, rows=None):
        q, k, v = projected
        b, tq = q.shape[:2]
        tk = k.shape[1]
        q, g = mx.split(q.reshape(b, tq, -1, 2 * self.d), 2, axis=-1)
        q_rope = tuple(r[:, -tq:] for r in rope)
        q = _rotate(self.q_norm(q), *q_rope).astype(v.dtype).transpose(0, 2, 1, 3)
        k = _rotate(self.k_norm(k.reshape(b, tk, -1, self.d)), *rope).astype(v.dtype).transpose(0, 2, 1, 3)
        v = v.reshape(b, tk, -1, self.d).transpose(0, 2, 1, 3)
        if state is not None:
            kp, vp = state if rows is None else tuple(a[rows] for a in state)
            k, v = mx.concatenate([kp, k], axis=2), mx.concatenate([vp, v], axis=2)
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return attn.transpose(0, 2, 1, 3).reshape(b, tq, -1) * mx.sigmoid(g.reshape(b, tq, -1)), (k, v)


class DecoderLayer:
    def __init__(self, layer, layer_type):
        self.gdn = layer_type == "linear_attention"
        self.mix = GatedDelta(layer.linear_attn) if self.gdn else Attention(layer.self_attn)
        self.pre = rms_norm(layer.input_layernorm)
        self.post = rms_norm(layer.post_attention_layernorm)
        self.up = Linear(layer.mlp.gate_proj, layer.mlp.up_proj)
        self.down = Linear(layer.mlp.down_proj, contiguous=True)

    def mlp(self, x, compute_dtype=None):
        streams = mx.split(self.up(self.post(x), compute_dtype), 2, axis=-1)
        return x + self.down(_swi_glu(*streams), compute_dtype)

    def stream(self, x, valid, rope, mask, save_state=False, compute_dtype=None):
        h = self.pre(x)
        if self.gdn:
            projected = self.mix.project(h * valid[..., None], compute_dtype)
            mix, state = self.mix.mix(projected, save_state=save_state)
        else:
            mix, state = self.mix.mix(self.mix.project(h, compute_dtype), rope, mask)
        return self.mlp(x + self.mix.out_proj(mix, compute_dtype), compute_dtype), state

    def branch(self, x, rope, mask, state, readout_only=False):
        if readout_only and not self.gdn:
            h = self.mix.project_readout(self.pre(x))
            x = x[:, -1:]
            mask = mask[..., -1:, :] if mask is not None else None
        else:
            h = self.mix.project(self.pre(x))
        batch = state[0].shape[0]
        if x.shape[0] != batch:
            h = tuple(mx.broadcast_to(t, (batch, *t.shape[1:])) for t in h)
        if self.gdn:
            mix, _ = self.mix.mix(h, state)
        else:
            mix, _ = self.mix.mix(h, rope, mask, state)
        return self.mlp(x + self.mix.out_proj(mix))

    def paired(self, xp, xq, valid, rope_p, rope_q, mask_p, mask_q, rows, compute_dtype=None):
        # Project both streams against the same weights in one GEMM.
        p, q = xp.shape[:2], xq.shape[:2]
        a, b = self.pre(xp), self.pre(xq)
        if self.gdn:
            a = a * valid[..., None]
        n = math.prod(p)
        packed = mx.concatenate([a.reshape(n, -1), b.reshape(math.prod(q), -1)])
        hp, hq = zip(*((z[:n].reshape(*p, -1), z[n:].reshape(*q, -1)) for z in self.mix.project(packed, compute_dtype)))
        if self.gdn:
            op, st = self.mix.mix(hp, save_state=True)
            oq, _ = self.mix.mix(hq, st, rows)
        else:
            op, st = self.mix.mix(hp, rope_p, mask_p)
            oq, _ = self.mix.mix(hq, rope_q, mask_q, st, rows)
        mixed = mx.concatenate([op.reshape(n, -1), oq.reshape(math.prod(q), -1)])
        residual = mx.concatenate([xp.reshape(n, -1), xq.reshape(math.prod(q), -1)])
        y = self.mlp(residual + self.mix.out_proj(mixed, compute_dtype), compute_dtype)
        return y[:n].reshape(xp.shape), y[n:].reshape(xq.shape)
