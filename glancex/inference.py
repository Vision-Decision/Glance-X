"""High-level inference API.

    from glancex import GlanceX
    m = GlanceX.from_pretrained("Vision-Decision/Glance-X-4B")           # or an exported folder, or a training .pt
    m.ask("park.jpg", ["the clear umbrella", "a red bicycle"])            # one image, many requests  (image_first)
    m.find("farm.jpg", "the two cows lying on the left", mode="joint")    # one image, one request    (joint)
    m.search("a black cow", ["a.jpg", "b.jpg", "c.jpg"])                  # one request, many images  (text_first)
"""
from dataclasses import dataclass, field

import torch
from PIL import Image

from .backbone import canonical_mode
from .build import build, load_ckpt, load_state
from .config import Config
from .data import collate, resize_dims
from .train import forward_batch, to_dev


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


def _default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class GlanceX:
    def __init__(self, net, heads, proc, cfg, device):
        self.net, self.heads, self.proc, self.cfg, self.device = net, heads, proc, cfg, device
        from .data import Tok
        self.tok = Tok(proc, cfg.max_q_tokens)

    @classmethod
    def from_pretrained(cls, src="Vision-Decision/Glance-X-4B", base=None, device=None, dtype=None, revision=None):
        """src: a Hugging Face repo id, an exported folder, or a training checkpoint (.pt).
        base: override the Qwen3.5 base (Hugging Face id or local path); default: the one in the config."""
        device = device or _default_device()
        dtype = dtype or (torch.bfloat16 if device == "cuda" else torch.float32)
        sd = load_state(src, revision)
        cfg = Config().update(sd["cfg"]).update(dict(grad_ckpt=False, pad_img_to=0, pad_txt_to=0, pad_txt_block_to=0,
                                                     pad_t2i_block_to=0, pad_bi_block_to=0))
        if base:
            cfg = cfg.update(dict(model_id=base))
        net, heads, proc = build(cfg, device, dtype=dtype)
        load_ckpt(sd, net, heads)
        net.eval(); heads.eval()
        return cls(net, heads, proc, cfg, device)

    # ------------------------------------------------------------------ helpers
    def _item(self, image, requests, max_vis_tokens=None):
        img = Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")
        W, H = img.size
        w, h = resize_dims(W, H, max_vis_tokens or self.cfg.max_vis_tokens, self.cfg.max_upscale)
        pv = self.proc.image_processor(images=[img.resize((w, h), Image.BICUBIC)], do_resize=False, return_tensors="pt")
        qs = [dict(text=t, present=True) for t in requests]
        return dict(pixel_values=pv["pixel_values"], grid_thw=pv["image_grid_thw"][0], qs=qs,
                    q_ids=[self.tok.text(t) for t in requests], present=[True] * len(qs), boxes=[[] for _ in qs],
                    rec={}, orig_wh=(W, H))

    @torch.no_grad()
    def _run(self, items, mode, obj_thr):
        b = collate(items, self.tok, self.cfg, canonical_mode(mode))
        meta = b.pop("meta")
        b = to_dev(b, self.device)
        out, _, (o, rv, rc) = forward_batch(self.net, self.heads, b, train=False)
        multi = self.heads.decode_multi(o["r"], o["vis"], rv, rc, o["vis_index"], out, thr=obj_thr)
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
                              iou_prob=float(piou[k]), boxes=(mb.float().cpu() * s).tolist(),
                              box_scores=ms.float().cpu().tolist(), image_index=int(q_img[k])))
        return res

    # ------------------------------------------------------------------ public API
    def ask(self, image, requests, mode="image_first", obj_thr=0.5, max_vis_tokens=None):
        """One image, many requests, answered in parallel. Returns one Result per request (input order)."""
        requests = [requests] if isinstance(requests, str) else list(requests)
        return self._run([self._item(image, requests, max_vis_tokens)], mode, obj_thr)

    def find(self, image, request, mode="joint", obj_thr=0.5, max_vis_tokens=None):
        """One image, one request. `joint` lets the image and the request read each other (most accurate)."""
        return self.ask(image, [request], mode=mode, obj_thr=obj_thr, max_vis_tokens=max_vis_tokens)[0]

    def search(self, request, images, mode="text_first", batch_size=8, obj_thr=0.5, sort=True, max_vis_tokens=None):
        """One request over many images. Returns one Result per image (`image_index` says which), ranked by
        confidence when sort=True."""
        res = []
        for i in range(0, len(images), batch_size):
            chunk = [self._item(im, [request], max_vis_tokens) for im in images[i:i + batch_size]]
            for r in self._run(chunk, mode, obj_thr):
                r.image_index += i
                res.append(r)
        return sorted(res, key=lambda r: -r.confidence) if sort else res
