"""CUDA-graph inference: the language model, heads and multi-box decoding are captured once per shape bucket and
replayed with one launch (the vision tower stays eager). Removes the per-kernel Python/launch overhead that
dominates small batches.

Shapes are made static by padding: image stream to cfg.pad_img_to (642 = 640 patches + 2), text stream to 44,
visual rows to 640, and the number of images / requests up to the next bucket (padding rows repeat the last real
image / request; their outputs are dropped). Needs max_vis_tokens <= 640 and requests of <= 41 tokens.
"""
import torch

from .backbone import canonical_mode
from .data import collate

BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256)
IMG_LEN, TXT_LEN, N_VIS = 642, 44, 640
INPUTS = ("img_ids", "img_valid", "img_mask", "img_pos", "txt_ids", "txt_valid", "txt_pos", "q_img", "vis_valid",
          "centers")


def bucket(n):
    for b in BUCKETS:
        if n <= b:
            return b
    return None


class GraphRunner:
    def __init__(self, net, heads, tok, cfg, device):
        self.net, self.heads, self.tok, self.device = net, heads, tok, device
        self.cfg = cfg.update(dict(pad_img_to=IMG_LEN, pad_txt_to=TXT_LEN))
        self.graphs = {}
        self.pool = None

    def supports(self, items):
        n_img = [int(it["grid_thw"].prod()) // 4 for it in items]
        n_txt = [len(q) for it in items for q in it["q_ids"]]
        return max(n_img) <= N_VIS and max(n_txt) <= TXT_LEN and bucket(len(items)) and bucket(sum(len(it["q_ids"]) for it in items))

    def _pad(self, items):
        """Repeat the last image / request up to the bucket sizes. -> padded items, number of real requests."""
        M = sum(len(it["q_ids"]) for it in items)
        items = list(items) + [items[-1]] * (bucket(len(items)) - len(items))
        Mb = bucket(sum(len(it["q_ids"]) for it in items))
        extra = Mb - sum(len(it["q_ids"]) for it in items)
        if extra:
            last = dict(items[-1])
            for k in ("q_ids", "qs", "present", "boxes"):
                last[k] = list(last[k]) + [last[k][-1]] * extra
            items[-1] = last
        return items, M

    def _core(self, s, mode, thr):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = self.net(s, s["feats"], mode)
        q_img = s["q_img"]
        if mode == "i2t":
            rv, rc = s["vis_valid"], s["centers"]
        else:
            rv, rc = s["vis_valid"][q_img], s["centers"][q_img]
        out = self.heads(o["r"], o["vis"], rv, rc, o["vis_index"], q_tok=o["q_tok"], q_valid=o["q_valid"], r_idx=o["r_idx"])
        mb, ms, mk = self.heads.multi_tensors(o["r"], o["vis"], rv, rc, o["vis_index"], out, thr=thr)
        return dict(p=out["presence_logit"].float().sigmoid(), piou=out["iou_logit"].float().sigmoid(),
                    box=out["box_pred"].float().clamp(0, 1), mb=mb, ms=ms, mk=mk)

    @torch.no_grad()
    def run(self, items, mode, thr):
        """-> (dict of CPU tensors for the real requests, meta, q_img)."""
        mode = canonical_mode(mode)
        items, M = self._pad(items)
        b = collate(items, self.tok, self.cfg, mode)
        meta = b.pop("meta")[:M]
        B = len(items)
        for k, w in (("vis_valid", N_VIS), ("centers", N_VIS)):              # visual rows to a fixed width
            x = b[k]
            pad = torch.zeros((x.shape[0], w - x.shape[1]) + tuple(x.shape[2:]), dtype=x.dtype)
            b[k] = torch.cat([x, pad], 1)
        feats = self.net.image_features(b["pixel_values"].to(self.device, non_blocking=True),
                                        b["grid_thw"].to(self.device))
        feats = torch.cat(feats, 0)
        key = (mode, B, len(b["q_img"]), thr)
        if key not in self.graphs:
            self.graphs[key] = self._capture(b, feats, mode, thr)
        g, s, out = self.graphs[key]
        for k in INPUTS:
            s[k].copy_(b[k], non_blocking=True)
        s["feats"][: feats.shape[0]].copy_(feats)
        g.replay()
        res = {k: v[:M].cpu() for k, v in out.items()}
        return res, meta, b["q_img"][:M]

    def _capture(self, b, feats, mode, thr):
        s = {k: b[k].to(self.device).clone() for k in INPUTS}
        s["feats"] = torch.zeros(len(s["img_ids"]) * N_VIS, feats.shape[1], dtype=feats.dtype, device=self.device)
        s["feats"][: feats.shape[0]].copy_(feats)
        side = torch.cuda.Stream(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(2):                                                # warm-up: autotune + allocator
                self._core(s, mode, thr)
        torch.cuda.current_stream(self.device).wait_stream(side)
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool, capture_error_mode="thread_local"):
            out = self._core(s, mode, thr)
        return g, s, out
