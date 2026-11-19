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

from ._box import BoxHeads, box_iou_giou, center_target_box, cramer, edge_targets


class GlanceXHeads(BoxHeads):
    def __init__(self, D, cfg):
        super().__init__(D, cfg)
        d = cfg.head_dim
        self.o_q = nn.Linear(D, d)
        self.o_k = nn.Linear(D, d)
        self.o_b = nn.Parameter(torch.tensor(-4.0))   # prior: most patches are not target centres

    def forward(self, hq, vis, vis_valid, centers, q_img, sel_idx=None):
        out = super().forward(hq, vis, vis_valid, centers, q_img, sel_idx=sel_idx)
        h = self.q_norm(hq.float())
        v = self.v_norm(vis.float())
        ol = torch.einsum("md,mnd->mn", self.o_q(h), self.o_k(v)[q_img]) / math.sqrt(self.o_q.out_features) + self.o_b
        out["obj_logits"] = ol.masked_fill(~vis_valid[q_img], -1e4)
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
        vi = q_img[:, None].expand(M, k)
        v_sel, pc = v[vi, idx], centers[vi, idx]                               # [M,k,D], [M,k,2]
        hk = h[:, None].expand(M, k, h.shape[-1])
        el = self.edges_logits(hk.reshape(M * k, -1), v_sel.reshape(M * k, -1), pc.reshape(M * k, 2))
        boxes = self.expected_box(el, pc.reshape(M * k, 2)).view(M, k, 4)
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
