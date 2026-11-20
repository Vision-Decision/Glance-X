"""Records -> batches of image streams and text streams (see glancex/backbone.py).

Record (one per image): image_id, file, W, H, questions[]; question: text, present, box (single target or null),
boxes (all targets, xyxy pixels; optional), orig (True = original human data), kind, lang.
"""
import json
import math
import random

import torch
from PIL import Image


def resize_dims(W, H, max_tokens, max_upscale, factor=32):
    """Largest (W', H'), multiples of `factor`, aspect ratio kept, with (W'/f) * (H'/f) <= max_tokens."""
    s = min(math.sqrt(max_tokens * factor * factor / (W * H)), max_upscale)
    while True:
        w = max(factor, int(round(W * s / factor)) * factor)
        h = max(factor, int(round(H * s / factor)) * factor)
        if (w // factor) * (h // factor) <= max_tokens:
            return w, h
        s *= 0.98


class Tok:
    """Image stream: <vision_start> <image_pad>*n <vision_end>.  Text stream: <ref> expr </ref> <box_start>."""

    def __init__(self, processor, max_q_tokens):
        self.tok = processor.tokenizer
        t = self.tok.convert_tokens_to_ids
        self.vs, self.ve, self.pad_img = t("<|vision_start|>"), t("<|vision_end|>"), t("<|image_pad|>")
        self.ors, self.ore, self.bs = t("<|object_ref_start|>"), t("<|object_ref_end|>"), t("<|box_start|>")
        for x in (self.vs, self.ve, self.pad_img, self.ors, self.ore, self.bs):
            assert isinstance(x, int) and x >= 0 and x != self.tok.unk_token_id, "special token missing"
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self.max_q = max_q_tokens

    def text(self, s):
        ids = self.tok(s, add_special_tokens=False)["input_ids"][: self.max_q]
        return [self.ors] + ids + [self.ore, self.bs]


def targets_of(q, W, H, max_targets):
    bs = q.get("boxes") or ([q["box"]] if q.get("box") is not None else [])
    if not q["present"]:
        bs = []
    return [[b[0] / W, b[1] / H, b[2] / W, b[3] / H] for b in bs[:max_targets]]


class RecordDataset(torch.utils.data.Dataset):
    def __init__(self, path, image_root, processor, cfg, train=True, limit=0):
        self.recs = [json.loads(l) for l in open(path)]
        if limit:
            self.recs = self.recs[:limit]
        self.root, self.proc, self.cfg, self.train = image_root, processor, cfg, train
        self.tok = Tok(processor, cfg.max_q_tokens)

    def __len__(self):
        return len(self.recs)

    def _pick(self, qs):
        n = self.cfg.max_q_per_record
        if not self.train or len(qs) <= n:
            return qs
        orig = [q for q in qs if q.get("orig")]
        aug = [q for q in qs if not q.get("orig")]
        n_orig = min(len(orig), max(int(round(n * self.cfg.min_orig_frac)), n - len(aug)))
        return random.sample(orig, n_orig) + random.sample(aug, min(len(aug), n - n_orig))

    def __getitem__(self, i):
        r = self.recs[i]
        img = Image.open(f"{self.root}/{r['file']}").convert("RGB")
        W, H = img.size
        w, h = resize_dims(W, H, self.cfg.max_vis_tokens, self.cfg.max_upscale)
        pv = self.proc.image_processor(images=[img.resize((w, h), Image.BICUBIC)], do_resize=False, return_tensors="pt")
        qs = self._pick(r["questions"])
        return dict(pixel_values=pv["pixel_values"], grid_thw=pv["image_grid_thw"][0], qs=qs,
                    q_ids=[self.tok.text(q["text"]) for q in qs], present=[bool(q["present"]) for q in qs],
                    boxes=[targets_of(q, W, H, self.cfg.max_targets) for q in qs], rec=r, orig_wh=(W, H))


def collate(items, tok, cfg, mode, q_per_image=0, merge=2):
    """mode: i2t | t2i | bi (stored in the batch; the layout is the same for all three).
    q_per_image > 0: exactly that many questions per image (sampled, with replacement if fewer) -> fixed shapes.
    0: every question of every item, in order (evaluation)."""
    sel = []
    for b, it in enumerate(items):
        n = len(it["qs"])
        if q_per_image:
            idx = random.sample(range(n), q_per_image) if n >= q_per_image else [random.randrange(n) for _ in range(q_per_image)]
        else:
            idx = list(range(n))
        sel += [(b, j) for j in idx]
    B, M = len(items), len(sel)
    grids = [it["grid_thw"] for it in items]
    dims = [(int(g[1] // merge), int(g[2] // merge)) for g in grids]
    n_img = [gh * gw for gh, gw in dims]
    N = max(n_img)
    Li = max(N + 2, cfg.pad_img_to)
    img_ids = torch.full((B, Li), tok.pad, dtype=torch.long)
    img_valid = torch.zeros(B, Li, dtype=torch.bool)
    img_mask = torch.zeros(B, Li, dtype=torch.bool)
    img_pos = torch.zeros(3, B, Li, dtype=torch.long)
    vis_valid = torch.zeros(B, N, dtype=torch.bool)
    centers = torch.zeros(B, N, 2)
    for b, ((gh, gw), n) in enumerate(zip(dims, n_img)):
        s = Li - (n + 2)
        img_ids[b, s] = tok.vs
        img_ids[b, s + 1: s + 1 + n] = tok.pad_img
        img_ids[b, s + 1 + n] = tok.ve
        img_valid[b, s:] = True
        img_mask[b, s + 1: s + 1 + n] = True
        ii, jj = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
        img_pos[0, b, s + 1: s + 1 + n] = 1
        img_pos[1, b, s + 1: s + 1 + n] = 1 + ii.flatten()
        img_pos[2, b, s + 1: s + 1 + n] = 1 + jj.flatten()
        img_pos[:, b, s + 1 + n] = 1 + max(gh, gw)
        assert 1 + max(gh, gw) < cfg.text_pos0, "image grid too large for text_pos0"
        vis_valid[b, :n] = True
        centers[b, :n, 0] = (jj.flatten() + 0.5) / gw
        centers[b, :n, 1] = (ii.flatten() + 0.5) / gh
    q_list = [items[b]["q_ids"][j] for b, j in sel]
    Lt = max(max(len(q) for q in q_list), cfg.pad_txt_to)
    txt_ids = torch.full((M, Lt), tok.pad, dtype=torch.long)
    txt_valid = torch.zeros(M, Lt, dtype=torch.bool)
    txt_pos = torch.zeros(3, M, Lt, dtype=torch.long)
    for m, q in enumerate(q_list):
        txt_ids[m, Lt - len(q):] = torch.tensor(q)
        txt_valid[m, Lt - len(q):] = True
        txt_pos[:, m, Lt - len(q):] = cfg.text_pos0 + torch.arange(len(q))
    tb = [items[b]["boxes"][j] for b, j in sel]
    T = max(1, max(len(x) for x in tb))
    boxes = torch.zeros(M, T, 4)
    n_boxes = torch.tensor([len(x) for x in tb])
    box_sel = torch.zeros(M, 4)
    for m, x in enumerate(tb):
        if x:
            boxes[m, :len(x)] = torch.tensor(x)
            box_sel[m] = torch.tensor(random.choice(x) if q_per_image else x[0])
    present = torch.tensor([items[b]["present"][j] for b, j in sel]) & (n_boxes > 0)
    return dict(
        pixel_values=torch.cat([it["pixel_values"] for it in items]), grid_thw=torch.stack(grids),
        img_ids=img_ids, img_valid=img_valid, img_mask=img_mask, img_pos=img_pos,
        txt_ids=txt_ids, txt_valid=txt_valid, txt_pos=txt_pos, q_img=torch.tensor([b for b, _ in sel]),
        vis_valid=vis_valid, centers=centers, grid_hw=torch.tensor(dims),
        present=present, boxes=boxes, n_boxes=n_boxes, box_sel=box_sel,
        n_tokens=int(img_valid.sum() + txt_valid.sum()), mode=mode,
        meta=[dict(rec=items[b]["rec"], q=items[b]["qs"][j], orig_wh=items[b]["orig_wh"]) for b, j in sel],
    )
