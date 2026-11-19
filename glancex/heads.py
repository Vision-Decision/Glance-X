"""Glance-X heads: the Glance single-box primitive plus a multi-box (per-patch objectness) branch.

  presence    R -> P(present)
  center      softmax over patches (one box: the most likely target)
  objectness  one sigmoid per patch: "this patch is near the centre of a target" (several boxes)
  edges       4 ordinal distributions at a patch -> box;  IoU head -> P(IoU >= 0.5)
All geometry is normalized [0,1]. Targets of a question: boxes [M,T,4] with n_boxes [M] (0 = absent).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._box import BoxHeads, box_iou_giou, center_target_box, cramer, edge_targets, fourier, mlp


class TwoWayLayer(nn.Module):
    """One SAM-style two-way block (pre-norm): request tokens self-attend, read the patches; patches read the
    request tokens. Only the patch side is long (N ~ 640), and every op on it is small (width d)."""

    def __init__(self, d, heads, mlp_ratio=2):
        super().__init__()
        A = lambda: nn.MultiheadAttention(d, heads, batch_first=True)  # noqa: E731
        self.q_self, self.q2v, self.v2q = A(), A(), A()
        self.n_qs, self.n_q2v, self.n_v, self.n_qm, self.n_v2q_v, self.n_v2q_q, self.n_vm = (nn.LayerNorm(d) for _ in range(7))
        self.q_mlp = mlp(d, mlp_ratio * d, d)
        self.v_mlp = mlp(d, mlp_ratio * d, d)

    def forward(self, q, v, q_pad, v_pad, pos):
        x = self.n_qs(q)
        q = q + self.q_self(x, x, x, key_padding_mask=q_pad, need_weights=False)[0]
        vn = self.n_v(v)
        q = q + self.q2v(self.n_q2v(q), vn + pos, vn, key_padding_mask=v_pad, need_weights=False)[0]
        q = q + self.q_mlp(self.n_qm(q))
        qn = self.n_v2q_q(q)
        v = v + self.v2q(self.n_v2q_v(v) + pos, qn, qn, key_padding_mask=q_pad, need_weights=False)[0]
        v = v + self.v_mlp(self.n_vm(v))
        return q, v


class TwoWayDecoder(nn.Module):
    """Light bidirectional fusion on top of the (query-independent, cacheable) image tokens and the request tokens.
    -> refined readout qr [M,d] and request-conditioned patch features vq [M,N,d]."""

    def __init__(self, D, d, n_layers, heads):
        super().__init__()
        self.in_v = nn.Linear(D, d)
        self.in_q = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.pos = nn.Linear(64, d)
        self.layers = nn.ModuleList(TwoWayLayer(d, heads) for _ in range(n_layers))
        self.out_q, self.out_v = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, v, q_img, v_valid, centers, q_tok, q_valid, r_idx):
        """v [R,N,D] normalised image tokens (rows = images in image_first); centers [R,N,2]; q_tok [M,T,D]."""
        vq = self.in_v(v)[q_img]                                              # project per image, then per request
        pos = self.pos(fourier(centers))[q_img]
        q = self.in_q(q_tok)
        q_pad, v_pad = ~q_valid, ~v_valid[q_img]
        for layer in self.layers:
            q, vq = layer(q, vq, q_pad, v_pad, pos)
        qr = self.out_q(q[torch.arange(q.shape[0], device=q.device), r_idx])
        return qr, self.out_v(vq)


def _zero_last(module):
    last = module[-1] if isinstance(module, nn.Sequential) else module
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    return module


class GlanceXHeads(BoxHeads):
    def __init__(self, D, cfg):
        super().__init__(D, cfg)
        d = cfg.head_dim
        self.o_q = nn.Linear(D, d)
        self.o_k = nn.Linear(D, d)
        self.o_b = nn.Parameter(torch.tensor(-4.0))   # prior: most patches are not target centres
        self.dec = None
        L = getattr(cfg, "decoder_layers", 0)
        if L:
            dd = cfg.decoder_dim
            self.dec = TwoWayDecoder(D, dd, L, cfg.decoder_heads)
            # residual heads on the decoder output; zero-initialised, so a new decoder starts as the identity
            self.dc_q, self.dc_k = nn.Linear(dd, d), _zero_last(nn.Linear(dd, d))
            self.do_q, self.do_k = nn.Linear(dd, d), _zero_last(nn.Linear(dd, d))
            self.dpres = _zero_last(mlp(dd, dd, 1))
            self.dedge = _zero_last(mlp(2 * dd + 64, 2 * dd, 4 * self.K))
            self.diou = _zero_last(mlp(2 * dd + 64, dd, 1))

    def _decode(self, v, q_img, vis_valid, centers, q_tok, q_valid, r_idx):
        if self.dec is None or q_tok is None:
            return None
        with torch.autocast(v.device.type, dtype=torch.bfloat16, enabled=v.device.type == "cuda"):
            qr, vq = self.dec(v, q_img, vis_valid, centers, q_tok, q_valid, r_idx)
        return qr.float(), vq.float()

    def _edges_at(self, h, v, q_img, centers, idx, dec):
        """Edge logits [M,k,4,K], IoU logits [M,k] and patch centres [M,k,2] at patches idx [M,k]."""
        M, k = idx.shape
        vi = q_img[:, None].expand(M, k)
        v_sel, pc = v[vi, idx], centers[vi, idx]
        f = torch.cat([h[:, None].expand(M, k, h.shape[-1]), v_sel, fourier(pc)], -1)
        el = self.edge(f).view(M, k, 4, self.K)
        iou = self.iou(f).squeeze(-1)
        if dec is not None:
            qr, vq = dec
            vq_sel = vq[torch.arange(M, device=vq.device)[:, None], idx]
            f2 = torch.cat([qr[:, None].expand(M, k, qr.shape[-1]), vq_sel, fourier(pc)], -1)
            el = el + self.dedge(f2).view(M, k, 4, self.K)
            iou = iou + self.diou(f2).squeeze(-1)
        return el, iou, pc

    def forward(self, hq, vis, vis_valid, centers, q_img, sel_idx=None, q_tok=None, q_valid=None, r_idx=None):
        """hq [M,D] readout; vis [R,N,D] image tokens (rows: images in image_first, requests otherwise); q_img [M]
        row of each request. q_tok/q_valid/r_idx: the request tokens for the two-way decoder (optional).
        Heads run in fp32 (the decoder in bf16 autocast)."""
        h = self.q_norm(hq.float())
        v = self.v_norm(vis.float())
        dec = self._decode(v, q_img, vis_valid, centers, q_tok, q_valid, r_idx)
        pres = self.presence(h).squeeze(-1)
        cl = torch.einsum("md,mnd->mn", self.c_q(h), self.c_k(v)[q_img]) / math.sqrt(self.c_q.out_features)
        ol = torch.einsum("md,mnd->mn", self.o_q(h), self.o_k(v)[q_img]) / math.sqrt(self.o_q.out_features) + self.o_b
        if dec is not None:
            qr, vq = dec
            pres = pres + self.dpres(qr).squeeze(-1)
            cl = cl + torch.einsum("md,mnd->mn", self.dc_q(qr), self.dc_k(vq)) / math.sqrt(self.dc_q.out_features)
            ol = ol + torch.einsum("md,mnd->mn", self.do_q(qr), self.do_k(vq)) / math.sqrt(self.do_q.out_features)
        valid = vis_valid[q_img]
        cl = cl.masked_fill(~valid, -1e4)
        pred_idx = cl.argmax(-1)
        el, iou, pc = self._edges_at(h, v, q_img, centers, pred_idx[:, None], dec)
        out = dict(presence_logit=pres, center_logits=cl, pred_idx=pred_idx, obj_logits=ol.masked_fill(~valid, -1e4),
                   edge_logits_pred=el[:, 0], box_pred=self.expected_box(el[:, 0], pc[:, 0]), iou_logit=iou[:, 0],
                   dec=dec)
        if sel_idx is not None:
            el_s, _, pc_s = self._edges_at(h, v, q_img, centers, sel_idx[:, None], dec)
            out.update(edge_logits=el_s[:, 0], pc_sel=pc_s[:, 0], box_sel=self.expected_box(el_s[:, 0], pc_s[:, 0]))
        return out

    @torch.no_grad()
    def decode_multi(self, hq, vis, vis_valid, centers, q_img, out, thr=0.5, topk=20, nms=0.5):
        """Boxes for every patch with objectness > thr (at most topk), then NMS. -> list of [k,4] tensors and scores."""
        boxes, sc, keep = self.multi_tensors(hq, vis, vis_valid, centers, q_img, out, thr, topk, nms)
        boxes, sc, keep = boxes.cpu(), sc.cpu(), keep.cpu()                    # one device->host copy
        return [(boxes[m][keep[m]], sc[m][keep[m]]) for m in range(boxes.shape[0])]

    @torch.no_grad()
    def multi_tensors(self, hq, vis, vis_valid, centers, q_img, out, thr=0.5, topk=20, nms=0.5):
        """decode_multi on the device with static shapes: boxes [M,k,4] (clamped), scores [M,k], keep [M,k]."""
        h = self.q_norm(hq.float())
        v = self.v_norm(vis.float())
        prob = out["obj_logits"].sigmoid()
        k = min(topk, prob.shape[1])
        sc, idx = prob.topk(k, dim=1)                                          # [M,k]
        M = h.shape[0]
        el, _, pc = self._edges_at(h, v, q_img, centers, idx, out.get("dec"))
        boxes = self.expected_box(el.reshape(M * k, 4, self.K), pc.reshape(M * k, 2)).view(M, k, 4)
        return boxes.clamp(0, 1), sc, greedy_nms(boxes, sc, thr, nms)


def greedy_nms(boxes, scores, thr, iou_thr):
    """Per-row greedy NMS, vectorised over rows. boxes [M,k,4], scores [M,k] sorted descending (topk order).
    A box is kept if its score > thr and its IoU with every higher-scored kept box is < iou_thr. -> keep [M,k]."""
    M, k = scores.shape
    iou = box_iou_giou(boxes[:, :, None], boxes[:, None, :])[0]               # [M,k,k]
    valid = scores > thr
    keep = torch.zeros_like(valid)
    for i in range(k):
        sup = (keep[:, :i] & (iou[:, i, :i] >= iou_thr)).any(1) if i else torch.zeros_like(valid[:, 0])
        keep[:, i] = valid[:, i] & ~sup
    return keep


def union_center_target(boxes, n_boxes, centers, valid, grid_hw):
    """Coverage of the union of all target boxes (normalized). boxes [M,T,4]."""
    M, T, _ = boxes.shape
    t = torch.zeros(M, centers.shape[1], device=boxes.device)
    for j in range(T):
        has = n_boxes > j
        if has.any():
            t[has] = t[has] + center_target_box(boxes[has, j], centers[has], valid[has], grid_hw[has]) * 1.0
    return t / t.sum(-1, keepdim=True).clamp(min=1e-12)


def obj_target(boxes, n_boxes, centers, valid, shrink=0.5):
    """Patch is positive if its centre lies in the central `shrink` part of any target box; every target keeps at
    least its nearest patch."""
    M, T, _ = boxes.shape
    t = torch.zeros(M, centers.shape[1], device=boxes.device)
    for j in range(T):
        has = n_boxes > j
        if not has.any():
            continue
        b = boxes[has, j]
        cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
        hw, hh = (b[:, 2] - b[:, 0]) * shrink / 2, (b[:, 3] - b[:, 1]) * shrink / 2
        c = centers[has]
        inside = ((c[..., 0] - cx[:, None]).abs() <= hw[:, None]) & ((c[..., 1] - cy[:, None]).abs() <= hh[:, None])
        d = (c[..., 0] - cx[:, None]) ** 2 + (c[..., 1] - cy[:, None]) ** 2
        d = d.masked_fill(~valid[has], 1e9)
        near = F.one_hot(d.argmin(-1), c.shape[1]).bool()
        t[has] = torch.maximum(t[has], (inside | near).float())
    return t * valid.float()


def sigmoid_focal(logits, target, valid, alpha=0.25, gamma=2.0):
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    w = alpha * target + (1 - alpha) * (1 - target)
    loss = (w * (1 - pt) ** gamma * ce) * valid.float()
    n_pos = target.sum(-1).clamp(min=1.0)
    return (loss.sum(-1) / n_pos).mean()


def loss_fn(out, tgt, cfg):
    """tgt: present [M], boxes [M,T,4], n_boxes [M], box_sel [M,4] (the target the edges are trained on),
    center_t [M,N] (union coverage), obj_t [M,N], valid [M,N]."""
    present = tgt["present"]
    L = {"presence": F.binary_cross_entropy_with_logits(out["presence_logit"], present.float())}
    L["obj"] = sigmoid_focal(out["obj_logits"], tgt["obj_t"], tgt["valid"])
    if present.any():
        pm = present
        L["center"] = -(tgt["center_t"][pm] * out["center_logits"][pm].log_softmax(-1)).sum(-1).mean()
        et, _ = edge_targets(tgt["box_sel"][pm], out["pc_sel"][pm], cfg, out["edge_logits"].shape[-1])
        L["cramer"] = cramer(out["edge_logits"][pm], et)
        _, giou = box_iou_giou(out["box_sel"][pm], tgt["box_sel"][pm])
        L["giou"] = (1 - giou).mean()
        with torch.no_grad():   # IoU of the decoded box with its best-matching target
            b = out["box_pred"][pm][:, None].expand(-1, tgt["boxes"].shape[1], 4)
            ious, _ = box_iou_giou(b, tgt["boxes"][pm])
            ious = ious.masked_fill(torch.arange(ious.shape[1], device=ious.device)[None] >= tgt["n_boxes"][pm][:, None], 0)
            best = ious.max(-1).values
        L["iou_head"] = F.binary_cross_entropy_with_logits(out["iou_logit"][pm], (best >= 0.5).float())
        L["_acc50"] = (best >= 0.5).float().mean().detach()
    else:
        z = out["presence_logit"].sum() * 0
        for k in ["center", "cramer", "giou", "iou_head"]:
            L[k] = z
    total = (cfg.w_presence * L["presence"] + cfg.w_center * L["center"] + cfg.w_cramer * L["cramer"]
             + cfg.w_giou * L["giou"] + cfg.w_iou_head * L["iou_head"] + cfg.w_obj * L["obj"])
    return total, L
