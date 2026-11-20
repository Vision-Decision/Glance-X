"""High-level inference API.

    from glancex import GlanceX
    m = GlanceX.from_pretrained("Vision-Decision/Glance-X-Lite")         # or an exported folder, or a training .pt
    m.ask("park.jpg", ["the clear umbrella", "a red bicycle"])            # one image, many requests  (image_first)
    bank = m.encode(photos); m.search("a black cow", bank)                # one request, cached images (image_first)
    m.search("a black cow", ["a.jpg", "b.jpg"], mode="text_first")        # one request, new images   (text_first)
"""
import copy
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from .backbone import canonical_mode
from .build import build, load_ckpt, load_state
from .config import Config
from .data import collate, resize_dims
from .forward import forward_batch, to_dev


@dataclass
class Result:
    request: str
    present_prob: float               # P(the request matches something in the image)
    box: list                         # most likely target, [x1, y1, x2, y2] in pixels of the input image
    iou_prob: float                   # P(IoU(box, target) >= 0.5)
    boxes: list = field(default_factory=list)          # every target found by the multi-box head, in pixels
    box_scores: list = field(default_factory=list)     # objectness of each of `boxes`
    image_index: int = 0              # which input image (for `search`)

    @property
    def confidence(self):
        """P(present) * P(IoU >= 0.5): a single number to threshold or rank on."""
        return self.present_prob * self.iou_prob

    @property
    def present(self):
        return self.present_prob >= 0.5

    def __repr__(self):
        b = [round(v) for v in self.box]
        return (f"Result({self.request!r}, present_prob={self.present_prob:.3f}, box={b}, "
                f"confidence={self.confidence:.3f}, n_boxes={len(self.boxes)})")


@dataclass
class ImageBank:
    """Cached image_first state of N images (GlanceX.encode). Searching it runs only the request's short block per
    image, not the vision tower or the image prefix."""
    data: dict            # states (per fusion layer), p_valid [N,Lp], vis [N,640,D]
    vis_valid: torch.Tensor
    centers: torch.Tensor
    orig_wh: list
    device: str

    def __len__(self):
        return len(self.orig_wh)

    def slice(self, a, b):
        st = [tuple(t[a:b] for t in s) for s in self.data["states"]]
        d = dict(states=st, p_valid=self.data["p_valid"][a:b], vis=self.data["vis"][a:b])
        return ImageBank(d, self.vis_valid[a:b], self.centers[a:b], self.orig_wh[a:b], self.device)

    @property
    def nbytes(self):
        ts = [t for s in self.data["states"] for t in s] + [self.data["vis"]]
        return sum(t.numel() * t.element_size() for t in ts)

    @staticmethod
    def concat(banks):
        st = [tuple(torch.cat(ts, 0) for ts in zip(*layer)) for layer in zip(*[b.data["states"] for b in banks])]
        d = dict(states=st, p_valid=torch.cat([b.data["p_valid"] for b in banks]),
                 vis=torch.cat([b.data["vis"] for b in banks]))
        return ImageBank(d, torch.cat([b.vis_valid for b in banks]), torch.cat([b.centers for b in banks]),
                         [wh for b in banks for wh in b.orig_wh], banks[0].device)


def _default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class GlanceX:
    def __init__(self, net, heads, proc, cfg, device, fast=True, devices=None, cuda_graphs=True):
        self.proc, self.cfg, self.device = proc, cfg, device
        from .data import Tok
        self.tok = Tok(proc, cfg.max_q_tokens)
        self.fast = fast
        self._pool = ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 4)) if fast else None
        if fast:
            merge_lora(net.hf)
        # one replica per GPU: (device, backbone, heads). Work is split across them and run concurrently.
        self.replicas = [(device, net, heads)]
        for d in (devices or [])[1:]:
            self.replicas.append((d, copy.deepcopy(net).to(d), copy.deepcopy(heads).to(d)))
        self._gpu_pool = ThreadPoolExecutor(max_workers=len(self.replicas)) if len(self.replicas) > 1 else None
        self.runners = None
        if fast and cuda_graphs:
            from .fast import GraphRunner
            self.runners = [GraphRunner(n, h, self.tok, cfg, d) for d, n, h in self.replicas]

    @property
    def net(self):
        return self.replicas[0][1]

    @property
    def heads(self):
        return self.replicas[0][2]

    @classmethod
    def from_pretrained(cls, src="Vision-Decision/Glance-X-Lite", base=None, device=None, dtype=None, revision=None,
                        fast=True, devices=None, cuda_graphs=True):
        """src: a Hugging Face repo id, an exported folder, or a training checkpoint (.pt).
        base: override the Qwen3.5 base (Hugging Face id or local path); default: the one in the config.
        fast: merge LoRA into the base weights and preprocess images on worker threads + the GPU (inference only).
        fast=False keeps the exact training-time computation (unmerged LoRA, CPU preprocessing).
        devices: several GPUs, e.g. ["cuda:0", "cuda:1"]; one replica each, batches are split across them.
        cuda_graphs: with fast, replay the language model + heads as CUDA graphs (captured once per shape bucket)."""
        if devices:
            device = devices[0]
        device = device or _default_device()
        on_cuda = str(device).startswith("cuda")
        dtype = dtype or (torch.bfloat16 if on_cuda else torch.float32)
        sd = load_state(src, revision)
        cfg = Config().update(sd["cfg"]).update(dict(grad_ckpt=False, pad_img_to=0, pad_txt_to=0, pad_txt_block_to=0,
                                                     pad_t2i_block_to=0, pad_bi_block_to=0))
        if base:
            cfg = cfg.update(dict(model_id=base))
        net, heads, proc = build(cfg, device, dtype=dtype)
        load_ckpt(sd, net, heads)
        net.eval(); heads.eval()
        return cls(net, heads, proc, cfg, device, fast=fast and on_cuda, devices=devices, cuda_graphs=cuda_graphs)

    # ------------------------------------------------------------------ helpers
    def _load(self, image, max_vis_tokens=None):
        """Decode + resize on the CPU (thread-safe). -> uint8 tensor [3,h,w], original (W, H)."""
        img = Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")
        W, H = img.size
        w, h = resize_dims(W, H, max_vis_tokens or self.cfg.max_vis_tokens, self.cfg.max_upscale)
        arr = np.array(img.resize((w, h), Image.BICUBIC))
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous(), (W, H)

    def _items(self, images, requests_per_image, max_vis_tokens=None, device=None):
        """Images -> collate items. fast: loading on worker threads, normalise + patchify on the GPU."""
        if not self.fast:
            return [self._item(im, rq, max_vis_tokens) for im, rq in zip(images, requests_per_image)]
        device = device or self.device
        loaded = list(self._pool.map(lambda im: self._load(im, max_vis_tokens), images))
        ts = [t.pin_memory().to(device, non_blocking=True) for t, _ in loaded]
        pv = self.proc.image_processor(images=ts, do_resize=False, return_tensors="pt", device=device)
        grids = pv["image_grid_thw"]
        n = grids.prod(-1).tolist()
        chunks = pv["pixel_values"].to(torch.bfloat16).split(n)
        return [self._make_item(px, g, rq, wh) for px, g, rq, (_, wh) in zip(chunks, grids, requests_per_image, loaded)]

    def _make_item(self, pixel_values, grid_thw, requests, orig_wh):
        qs = [dict(text=t, present=True) for t in requests]
        return dict(pixel_values=pixel_values, grid_thw=grid_thw, qs=qs, q_ids=[self.tok.text(t) for t in requests],
                    present=[True] * len(qs), boxes=[[] for _ in qs], rec={}, orig_wh=orig_wh)

    def _item(self, image, requests, max_vis_tokens=None):
        img = Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")
        W, H = img.size
        w, h = resize_dims(W, H, max_vis_tokens or self.cfg.max_vis_tokens, self.cfg.max_upscale)
        pv = self.proc.image_processor(images=[img.resize((w, h), Image.BICUBIC)], do_resize=False, return_tensors="pt")
        return self._make_item(pv["pixel_values"], pv["image_grid_thw"][0], requests, (W, H))

    def _check_mode(self, mode):
        """Only the directions the checkpoint was trained in (Glance-X-Lite: image_first and text_first)."""
        w = [float(x) for x in self.cfg.mode_probs.split(",")]
        m = canonical_mode(mode)
        if w[("i2t", "t2i", "bi").index(m)] <= 0:
            raise ValueError(f"this checkpoint was not trained in the {mode!r} direction")
        return m

    @torch.no_grad()
    def _run(self, items, mode, obj_thr, replica=0):
        mode = self._check_mode(mode)
        device, net, heads = self.replicas[replica]
        runner = self.runners[replica] if self.runners else None
        if runner is not None and runner.supports(items):
            r, meta, q_img = runner.run(items, mode, obj_thr)
            multi = [(r["mb"][k][r["mk"][k]], r["ms"][k][r["mk"][k]]) for k in range(len(meta))]
            p, piou, box = r["p"], r["piou"], r["box"]
        else:
            b = collate(items, self.tok, self.cfg, canonical_mode(mode))
            meta = b.pop("meta")
            b = to_dev(b, device)
            out, _, (o, rv, rc) = forward_batch(net, heads, b, train=False)
            multi = heads.decode_multi(o["r"], o["vis"], rv, rc, o["vis_index"], out, thr=obj_thr)
            p = out["presence_logit"].float().sigmoid().cpu()
            piou = out["iou_logit"].float().sigmoid().cpu()
            box = out["box_pred"].float().clamp(0, 1).cpu()
            q_img = b["q_img"].cpu()
        res = []
        for k, mt in enumerate(meta):
            W, H = mt["orig_wh"]
            s = torch.tensor([W, H, W, H], dtype=torch.float)
            mb, ms = multi[k]
            res.append(Result(request=mt["q"]["text"], present_prob=float(p[k]), box=(box[k] * s).tolist(),
                              iou_prob=float(piou[k]), boxes=(mb.float() * s).tolist(),
                              box_scores=ms.float().tolist(), image_index=int(q_img[k])))
        return res

    def _sharded(self, n, fn):
        """Split range(n) into one contiguous shard per replica, run fn(replica, start, stop) concurrently."""
        R = min(len(self.replicas), n)
        cuts = [round(i * n / R) for i in range(R + 1)]
        jobs = [(r, cuts[r], cuts[r + 1]) for r in range(R)]

        def run(job):
            r, a, b = job
            dev = self.replicas[r][0]
            if not str(dev).startswith("cuda"):
                return fn(r, a, b)
            with torch.cuda.device(dev):
                return fn(r, a, b)
        if R == 1:
            return run(jobs[0])
        return [x for part in self._gpu_pool.map(run, jobs) for x in part]

    # ------------------------------------------------------------------ public API
    def ask(self, image, requests, mode="image_first", obj_thr=0.5, max_vis_tokens=None, min_split=32):
        """One image, many requests, answered in parallel. Returns one Result per request (input order).
        With several GPUs and >= min_split requests, the requests are split across them."""
        requests = [requests] if isinstance(requests, str) else list(requests)
        if len(self.replicas) == 1 or len(requests) < min_split:
            return self._run(self._items([image], [requests], max_vis_tokens), mode, obj_thr)
        loaded = self._items([image], [requests], max_vis_tokens)[0] if self.fast else None

        def part(r, a, b):
            dev = self.replicas[r][0]
            if loaded is not None:
                it = self._make_item(loaded["pixel_values"].to(dev, non_blocking=True), loaded["grid_thw"],
                                     requests[a:b], loaded["orig_wh"])
            else:
                it = self._item(image, requests[a:b], max_vis_tokens)
            return self._run([it], mode, obj_thr, replica=r)
        return self._sharded(len(requests), part)

    # ------------------------------------------------------------------ image cache
    @torch.no_grad()
    def encode(self, images, batch_size=32):
        """Encode images once (image_first prefix, request-independent) -> ImageBank for fast search()."""
        from .fast import IMG_LEN, N_VIS
        cfg = self.cfg.update(dict(pad_img_to=IMG_LEN))
        device, net, _ = self.replicas[0]
        banks = []
        for i in range(0, len(images), batch_size):
            items = self._items(images[i:i + batch_size], [["x"]] * len(images[i:i + batch_size]), device=device)
            b = to_dev({k: v for k, v in collate(items, self.tok, cfg, "i2t").items() if k != "meta"}, device)
            assert b["img_ids"].shape[1] == IMG_LEN, "encode() supports max_vis_tokens <= 640"
            feats = net.image_features(b["pixel_values"], b["grid_thw"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                d = net.encode_images(b, feats)
            n = b["vis_valid"].shape[1]
            pad = lambda t: torch.cat([t, t.new_zeros((t.shape[0], N_VIS - n) + tuple(t.shape[2:]))], 1)  # noqa: E731
            d["vis"] = pad(d["vis"])
            banks.append(ImageBank(d, pad(b["vis_valid"]), pad(b["centers"]), [it["orig_wh"] for it in items], device))
        return ImageBank.concat(banks) if len(banks) > 1 else banks[0]

    @torch.no_grad()
    def _search_bank(self, request, bank, obj_thr, chunk=256):
        device, net, heads = self.replicas[0]
        ids = self.tok.text(request)
        L = max(len(ids), self.cfg.pad_txt_to, 1)
        txt = torch.full((1, L), self.tok.pad, dtype=torch.long)
        txt[0, L - len(ids):] = torch.tensor(ids)
        tv = torch.zeros(1, L, dtype=torch.bool)
        tv[0, L - len(ids):] = True
        tpos = torch.zeros(3, 1, L, dtype=torch.long)
        tpos[:, 0, L - len(ids):] = self.cfg.text_pos0 + torch.arange(len(ids))
        tb = to_dev(dict(txt_ids=txt, txt_valid=tv, txt_pos=tpos), device)
        res = []
        for a in range(0, len(bank), chunk):
            bk = bank.slice(a, min(len(bank), a + chunk))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = net.query_bank(bk.data, tb)
            out = heads(o["r"], o["vis"], bk.vis_valid, bk.centers, o["vis_index"])
            multi = heads.decode_multi(o["r"], o["vis"], bk.vis_valid, bk.centers, o["vis_index"], out, thr=obj_thr)
            p, piou = out["presence_logit"].float().sigmoid().cpu(), out["iou_logit"].float().sigmoid().cpu()
            box = out["box_pred"].float().clamp(0, 1).cpu()
            for k, (W, H) in enumerate(bk.orig_wh):
                s = torch.tensor([W, H, W, H], dtype=torch.float)
                mb, ms = multi[k]
                res.append(Result(request=request, present_prob=float(p[k]), box=(box[k] * s).tolist(),
                                  iou_prob=float(piou[k]), boxes=(mb.float() * s).tolist(),
                                  box_scores=ms.float().tolist(), image_index=a + k))
        return res

    def find(self, image, request, mode="image_first", obj_thr=0.5, max_vis_tokens=None):
        """One image, one request."""
        return self.ask(image, [request], mode=mode, obj_thr=obj_thr, max_vis_tokens=max_vis_tokens)[0]

    def search(self, request, images, mode="text_first", batch_size=64, obj_thr=0.5, sort=True, max_vis_tokens=None):
        """One request over many images. Returns one Result per image (`image_index` says which), ranked by
        confidence when sort=True. With several GPUs the images are split across them and run concurrently.
        images can be an ImageBank (from encode()): then only the request's block runs per image (image_first)."""
        if isinstance(images, ImageBank):
            res = self._search_bank(request, images, obj_thr)
            return sorted(res, key=lambda r: -r.confidence) if sort else res
        def part(r, a, b):
            out = []
            for i in range(a, b, batch_size):
                j = min(b, i + batch_size)
                chunk = self._items(images[i:j], [[request]] * (j - i), max_vis_tokens, device=self.replicas[r][0])
                for x in self._run(chunk, mode, obj_thr, replica=r):
                    x.image_index += i
                    out.append(x)
            return out
        res = self._sharded(len(images), part)
        return sorted(res, key=lambda r: -r.confidence) if sort else res


def merge_lora(model):
    """Fold every LoRA adapter into its base Linear (W += B A * scale, in the base dtype) and drop the wrapper."""
    for name, mod in list(model.named_modules()):
        if hasattr(mod, "lora_A") and hasattr(mod, "base_layer"):
            mod.merge()
            parent, _, attr = name.rpartition(".")
            setattr(model.get_submodule(parent) if parent else model, attr, mod.base_layer)
    return model
