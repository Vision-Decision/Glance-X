"""Apple-silicon backend for the two Glance-X-Lite inference directions."""

import math
from pathlib import Path
from statistics import median
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from ._mac_layers import DecoderLayer, Linear, Vision, array, layer_norm, rms_norm
from .data import resize_dims
from .inference import ImageBank, Result, merge_lora


class MacImageBank(ImageBank):
    """MLX cache. DeltaNet state uses fp32 and attention keys/values use fp16."""

    def slice(self, a, b):
        d = self.data
        states = [tuple(t[a:b] for t in layer) for layer in d["states"]]
        data = {k: v[a:b] for k, v in d.items() if k != "states"}
        return MacImageBank(dict(data, states=states), self.vis_valid[a:b], self.centers[a:b], self.orig_wh[a:b], self.device)

    @property
    def nbytes(self):
        return sum(t.nbytes for layer in self.data["states"] for t in layer) + sum(
            t.nbytes for key, t in self.data.items() if key != "states"
        ) + self.vis_valid.nbytes + self.centers.nbytes

    @staticmethod
    def concat(banks):
        if len(banks) == 1:
            return banks[0]

        def join(items, axis=None):
            if axis is None:
                return mx.concatenate(items)
            length = max(a.shape[axis] for a in items)
            pads = [[(0, length - s) if i == axis else (0, 0) for i, s in enumerate(a.shape)] for a in items]
            return mx.concatenate([mx.pad(a, p) if a.shape[axis] < length else a for a, p in zip(items, pads)])

        states = []
        for layer in zip(*(bank.data["states"] for bank in banks)):
            a, b = zip(*layer)
            axis = 2 if b[0].ndim == 4 else None
            states.append((join(a, axis), join(b, axis)))
        data = {k: join([bank.data[k] for bank in banks], 1) for k in banks[0].data if k != "states"}
        data["states"] = states
        valid = join([bank.vis_valid for bank in banks], 1)
        centers = join([bank.centers for bank in banks], 1)
        mx.eval(data, valid, centers)
        return MacImageBank(data, valid, centers, [wh for bank in banks for wh in bank.orig_wh], "mlx")


class BoxDecoder:
    def __init__(self, heads):
        self.q_norm, self.v_norm = layer_norm(heads.q_norm), layer_norm(heads.v_norm)
        self.presence = self.mlp(heads.presence)
        self.edge, self.iou = self.mlp(heads.edge), self.mlp(heads.iou)
        self.query = Linear(heads.c_q, heads.o_q, dtype=mx.float32)
        self.key = Linear(heads.c_k, heads.o_k, dtype=mx.float32)
        self.scale = heads.c_q.out_features ** -0.5
        self.bias = array(heads.o_b)
        self.bins = array(heads.bin_centers)
        self.frequencies = mx.array((2 ** np.arange(16, dtype=np.float32)) * np.float32(math.pi))

    @staticmethod
    def mlp(layers):
        up, down = (Linear(layers[i], dtype=mx.float32) for i in (0, 2))
        return lambda x: down(nn.gelu(up(x)))

    def prepare(self, vis):
        vis = self.v_norm(vis.astype(mx.float32))
        ck, ok = mx.split(self.key(vis), 2, axis=-1)
        return dict(vis=vis, center_keys=ck, object_keys=ok)

    def decode(self, readout, vis, centers, valid, rows):
        h = self.q_norm(readout)
        cq, oq = mx.split(self.query(h), 2, axis=-1)
        ck, ok = vis["center_keys"], vis["object_keys"]
        if rows is not None:
            ck, ok, valid = ck[rows], ok[rows], valid[rows]
        cl = mx.where(valid, (ck @ cq[..., None]).squeeze(-1) * self.scale, -1e4)
        ol = mx.where(valid, (ok @ oq[..., None]).squeeze(-1) * self.scale + self.bias, -1e4)
        first = mx.argmax(cl, axis=1)[:, None]
        top = mx.argsort(-ol, axis=1)[:, :min(20, ol.shape[1])]
        idx = mx.concatenate([first, top], axis=1)
        if rows is None:
            rows = mx.arange(h.shape[0])
        f, pc = vis["vis"][rows[:, None], idx], centers[rows[:, None], idx]
        angles = pc[..., None] * self.frequencies
        fourier = mx.concatenate([mx.sin(angles), mx.cos(angles)], axis=-1).reshape(*idx.shape, -1)
        features = mx.concatenate([mx.broadcast_to(h[:, None], (*idx.shape, h.shape[-1])), f, fourier], axis=-1)
        edges = self.edge(features).reshape(*idx.shape, 4, -1)
        offsets = mx.sum(mx.softmax(edges, axis=-1) * self.bins, axis=-1)
        boxes = mx.concatenate([pc - offsets[..., :2], pc + offsets[..., 2:]], axis=-1)
        p = mx.sigmoid(self.presence(h).squeeze(-1))
        piou = mx.sigmoid(self.iou(features[:, 0]).squeeze(-1))
        scores = mx.sigmoid(mx.take_along_axis(ol, top, axis=1))
        return p, piou, boxes, scores, cl


def _nms(boxes, scores, threshold):
    # The Torch API computes IoU before clamping the coordinates.
    a, b = boxes[:, None], boxes[None, :]
    wh = np.maximum(np.minimum(a[..., 2:], b[..., 2:]) - np.maximum(a[..., :2], b[..., :2]), 0)
    intersection = wh.prod(axis=-1)
    area = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=-1)
    iou = intersection / np.maximum(area[:, None] + area[None, :] - intersection, 1e-9)
    kept = []
    for i, score in enumerate(scores):
        if score <= threshold:
            break
        if not np.any(iou[i, kept] >= 0.5):
            kept.append(i)
    return kept


def _causal(valid):
    pos = mx.arange(valid.shape[1])
    return (((pos[:, None] >= pos) & valid[:, None, :]) | (pos[:, None] == pos))[:, None]


def _block_mask(prefix, valid, bidir=True):
    rows, length = valid.shape
    pos = mx.arange(length)
    within = valid[:, None, :] & (pos[:, None] >= pos if not bidir else mx.ones((length, length), mx.bool_))
    within = within | (pos[:, None] == pos)
    return mx.concatenate([mx.broadcast_to(prefix[:, None], (rows, length, prefix.shape[1])), within], axis=-1)[:, None]


class MacInference:
    def __init__(self, net, heads, processor, cfg, tok):
        merge_lora(net.hf)
        self.cfg, self.tok = cfg, tok
        self.vision = Vision(net.visual, processor.image_processor)
        self.box = BoxDecoder(heads)
        self.embedding = array(net.lm.embed_tokens.weight, mx.float16)
        self.norm = rms_norm(net.lm.norm)
        self.layers = [DecoderLayer(layer, name) for layer, name in zip(net.lm.layers, net.layer_types)]
        self.uni, self.fusion = self.layers[:net.K], self.layers[net.K:]
        rotary = net.lm.rotary_emb
        self.inv = array(rotary.inv_freq) * rotary.attention_scaling
        axes = np.zeros(len(rotary.inv_freq), dtype=np.int32)
        for axis in (1, 2):
            axes[axis:rotary.mrope_section[axis] * 3:3] = axis
        self.axes = mx.array(axes)
        self.geometry = {}
        self._timings = None
        self.query_layers = mx.compile(self._query_layers)
        self.paired_layers = mx.compile(self._paired_layers)
        self.image_layers = mx.compile(self._image_layers)

    def _stage(self, name, fn, *args):
        if self._timings is None:
            return fn(*args)
        start = perf_counter()
        result = fn(*args)
        mx.eval(result)
        self._timings[name] = self._timings.get(name, 0.0) + (perf_counter() - start) * 1000
        return result

    def _load(self, image, max_tokens):
        if isinstance(image, (str, Path)):
            with Image.open(image) as f:
                img = f.convert("RGB")
        else:
            img = image.convert("RGB")
        wh = img.size
        w, h = resize_dims(*wh, max_tokens or self.cfg.max_vis_tokens, self.cfg.max_upscale)
        pixels = mx.array(np.asarray(img.resize((w, h), Image.BICUBIC)))
        return pixels, wh

    def _rope(self, positions):
        freq = positions[..., self.axes] * self.inv
        freq = mx.concatenate([freq, freq], axis=-1)[:, :, None, :]
        return mx.cos(freq), mx.sin(freq)

    def _geometry(self, pixels):
        step = self.vision.patch_size * self.vision.merge
        gh, gw = (size // step for size in pixels.shape[:2])
        if (gh, gw) not in self.geometry:
            y, x = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
            n = gh * gw
            pos = np.zeros((1, n + 2, 3), dtype=np.int32)
            pos[0, 1:-1] = np.stack([np.ones(n), 1 + y.ravel(), 1 + x.ravel()], axis=-1)
            pos[0, -1] = 1 + max(gh, gw)
            assert pos.max() < self.cfg.text_pos0, "image grid too large for text_pos0"
            centers = np.stack([(x.ravel() + 0.5) / gw, (y.ravel() + 0.5) / gh], axis=-1).astype(np.float32)
            values = mx.array(pos), mx.array(centers[None])
            mx.eval(values)
            self.geometry[gh, gw] = values
        return self.geometry[gh, gw]

    def _image(self, pixels, features):
        pos, centers = self._geometry(pixels)
        start_end = self.embedding[mx.array([self.tok.vs, self.tok.ve])].astype(mx.float32)
        hidden = mx.concatenate([start_end[:1], features, start_end[1:]], axis=0)[None]
        valid = mx.ones(hidden.shape[:2], mx.bool_)
        rope = self._rope(pos)
        return self._unimodal(hidden, valid, rope, "causal"), pos, centers, valid, rope

    def _text(self, requests):
        ids = [self.tok.text(req) for req in requests]
        length = max(map(len, ids))
        tokens = np.full((len(ids), length), self.tok.pad, dtype=np.int32)
        pos = np.zeros(tokens.shape, dtype=np.int32)
        counts = np.array([len(row) for row in ids], dtype=np.int32)
        for i, row in enumerate(ids):
            tokens[i, -len(row):] = row
            pos[i, -len(row):] = self.cfg.text_pos0 + np.arange(len(row))
        hidden = self.embedding[mx.array(tokens)].astype(mx.float32)
        valid = mx.array(pos != 0)
        positions = mx.repeat(mx.array(pos)[..., None], 3, axis=-1)
        mask = _causal(valid)
        hidden = self._unimodal(hidden, valid, self._rope(positions), mask)
        return hidden, positions, valid, counts

    def _unimodal(self, x, valid, rope, mask):
        for layer in self.uni:
            x, _ = layer.stream(x, valid, rope, mask)
        return x

    def _encode(self, im):
        x, _, centers, valid, rope = im
        x, states = self.image_layers(x, valid, rope)
        data = dict(self.box.prepare(self.norm(x[:, 1:-1])), states=states, p_valid=valid)
        return data, centers, valid[:, 1:-1]

    def _image_layers(self, x, valid, rope):
        states = []
        for layer in self.fusion:
            x, state = layer.stream(x, valid, rope, "causal", save_state=True)
            states.append(state)
        return x, states

    def _paired_layers(self, xp, xq, p_valid, p_rope, q_rope, p_mask, q_mask, rows):
        for layer in self.fusion:
            xp, xq = layer.paired(xp, xq, p_valid, p_rope, q_rope, p_mask, q_mask, rows)
        return xp, xq

    def _query_layers(self, x, rope, mask, states):
        for layer, state in zip(self.fusion, states):
            x = layer.branch(x, rope, mask, state)
        return self.norm(x[:, -1])

    def _fuse(self, image, text, mode):
        xi, ip, _, iv, ir = image
        xt, tp, tv, counts = text
        batch, length = xt.shape[:2]
        rows = mx.zeros((batch,), mx.uint32)
        bidir = self.cfg.bidir_block_mask == "block_bidir"
        if mode == "i2t":
            pi = mx.arange(length)[None] + mx.array(length - counts)[:, None]
            xq = mx.take_along_axis(xt, mx.minimum(pi, length - 1)[..., None], axis=1)
            pos = self.cfg.text_pos0 + mx.arange(length)[None, :, None]
            qr = self._rope(mx.broadcast_to(pos, (batch, length, 3)))
            qv = mx.arange(length)[None] < mx.array(counts)[:, None]
            all_valid = bool(np.all(counts == length))
            mask = None if all_valid and bidir else _block_mask(iv[rows], qv, bidir)
            xi, xq = self.paired_layers(xi, xq, iv, ir, qr, "causal", mask, rows)
            return self.norm(xq[mx.arange(batch), mx.array(counts - 1)]), self.norm(xi[:, 1:-1]), rows
        xp, pv = xt[:, :-1], tv[:, :-1]
        xq = mx.concatenate([mx.broadcast_to(xi, (batch, *xi.shape[1:])), xt[:, -1:]], axis=1)
        pos = mx.concatenate([mx.broadcast_to(ip, (batch, *ip.shape[1:])), tp[:, -1:]], axis=1)
        qr, pr = self._rope(pos), self._rope(tp[:, :-1])
        pm = _causal(pv)
        qm = None if bool(np.all(counts == length)) and bidir else _block_mask(pv, mx.ones(xq.shape[:2], mx.bool_), bidir)
        xp, xq = self.paired_layers(xp, xq, pv, pr, qr, pm, qm, None)
        return self.norm(xq[:, -1]), self.norm(xq[:, 1:-2]), rows

    def _results(self, decoded, wh, requests, threshold, start=0, diagnostics=False, per_image=False):
        p, piou, boxes, scores, logits = decoded
        mx.eval(p, piou, boxes, scores)
        p, piou, boxes, scores = map(np.asarray, (p, piou, boxes, scores))
        result = []
        for i, (w, h) in enumerate(wh):
            b = boxes[i]
            keep = _nms(b[1:], scores[i], threshold)
            px = b.clip(0, 1) * np.array([w, h, w, h], dtype=np.float32)
            result.append(Result(requests[i], float(p[i]), px[0].tolist(), float(piou[i]),
                                 px[1:][keep].tolist(), scores[i][keep].tolist(), start + i if per_image else 0))
        if diagnostics:
            logits = np.asarray(logits)
            top = np.argsort(-logits, axis=1)[:, :2]
            return list(zip(result, (
                dict(center_indices=idx.tolist(), center_logits=row[idx].tolist()) for idx, row in zip(top, logits)
            )))
        return result

    def ask(self, image, requests, mode, threshold, max_tokens=None, diagnostics=False):
        if not requests:
            return []
        pixels, wh = self._stage("preprocess", self._load, image, max_tokens)
        features = self._stage("vision", self.vision, pixels)
        im = self._stage("image modality", self._image, pixels, features)
        out = []
        chunk = 8 if mode == "t2i" else 64
        for a in range(0, len(requests), chunk):
            req = requests[a:a+chunk]
            txt = self._stage("text modality", self._text, req)
            read, vis, rows = self._stage("fusion", self._fuse, im, txt, mode)
            centers, valid = im[2], im[3][:, 1:-1]
            if mode == "t2i":
                centers = mx.broadcast_to(centers, (len(req), *centers.shape[1:]))
                valid = mx.broadcast_to(valid, (len(req), valid.shape[1]))
                rows = None
            decoded = self._stage("heads", lambda: self.box.decode(read, self.box.prepare(vis), centers, valid, rows))
            start = perf_counter()
            out.extend(self._results(decoded, [wh] * len(req), req, threshold, diagnostics=diagnostics))
            if self._timings is not None:
                self._timings["result conversion"] = self._timings.get("result conversion", 0) + (perf_counter() - start) * 1000
        return out

    def encode(self, images, batch_size):
        banks = []
        for a in range(0, len(images), batch_size):
            for image in images[a:a+batch_size]:
                pixels, wh = self._load(image, None)
                d, centers, valid = self._encode(self._image(pixels, self.vision(pixels)))
                mx.eval(d, centers, valid)
                banks.append(MacImageBank(d, valid, centers, [wh], "mlx"))
        if not banks:
            raise ValueError("encode requires at least one image")
        return MacImageBank.concat(banks)

    def search_bank(self, request, bank, threshold, diagnostics=False):
        xt, tp, _, _ = self._text([request])
        qr = self._rope(tp)
        result = []
        for a in range(0, len(bank), 256):
            part = bank if a == 0 and len(bank) <= 256 else bank.slice(a, a + 256)
            data = part.data
            x = mx.broadcast_to(xt, (len(part), *xt.shape[1:]))
            # Every cached row can have its own length.
            mask = _block_mask(data["p_valid"], mx.ones(x.shape[:2], mx.bool_), self.cfg.bidir_block_mask == "block_bidir")
            readout = self.query_layers(x, qr, mask, data["states"])
            decoded = self.box.decode(readout, data, part.centers, part.vis_valid, None)
            result.extend(self._results(
                decoded, part.orig_wh, [request] * len(part), threshold, a, diagnostics=diagnostics, per_image=True,
            ))
        return result

    def profile(self, image, request, mode, repeats):
        from .backbone import canonical_mode

        mode = canonical_mode(mode)
        self.ask(image, [request], mode, 0.5)
        samples = []
        try:
            for _ in range(repeats):
                self._timings = {}
                start = perf_counter()
                self.ask(image, [request], mode, 0.5)
                mx.synchronize()
                self._timings["total"] = (perf_counter() - start) * 1000
                samples.append(self._timings)
        finally:
            self._timings = None
        return {k: median(s[k] for s in samples) for k in samples[0]}
