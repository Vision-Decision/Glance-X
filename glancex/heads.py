"""Glance-X heads: the Glance single-box primitive plus a multi-box (per-patch objectness) branch (inference).

  presence    R -> P(present)
  center      softmax over patches (one box: the most likely target)
  objectness  one sigmoid per patch: "this patch is near the centre of a target" (several boxes)
  edges       4 ordinal distributions at a patch -> box;  IoU head -> P(IoU >= 0.5)
All geometry is normalized [0,1]. Targets of a question: boxes [M,T,4] with n_boxes [M] (0 = absent).
"""
import math
import torch
import torch.nn as nn

from ._box import BoxHeads, box_iou_giou, fourier


class GlanceXHeads(BoxHeads):
    def __init__(self, D, cfg):
        super().__init__(D, cfg)
        d = cfg.head_dim
        self.o_q = nn.Linear(D, d)
        self.o_k = nn.Linear(D, d)
        self.o_b = nn.Parameter(torch.tensor(-4.0))   # prior: most patches are not target centres

    def _edges_at(self, h, v, q_img, centers, idx):
        """Edge logits [M,k,4,K], IoU logits [M,k] and patch centres [M,k,2] at patches idx [M,k]."""
        M, k = idx.shape
        vi = q_img[:, None].expand(M, k)
        v_sel, pc = v[vi, idx], centers[vi, idx]
        f = torch.cat([h[:, None].expand(M, k, h.shape[-1]), v_sel, fourier(pc)], -1)
        return self.edge(f).view(M, k, 4, self.K), self.iou(f).squeeze(-1), pc

    def forward(self, hq, vis, vis_valid, centers, q_img):
        """hq [M,D] readout; vis [R,N,D] image tokens (rows: images in image_first, requests in text_first);
        q_img [M] row of each request. Heads run in fp32."""
        h = self.q_norm(hq.float())
        v = self.v_norm(vis.float())
        pres = self.presence(h).squeeze(-1)
        cl = torch.einsum("md,mnd->mn", self.c_q(h), self.c_k(v)[q_img]) / math.sqrt(self.c_q.out_features)
        ol = torch.einsum("md,mnd->mn", self.o_q(h), self.o_k(v)[q_img]) / math.sqrt(self.o_q.out_features) + self.o_b
        valid = vis_valid[q_img]
        cl = cl.masked_fill(~valid, -1e4)
        pred_idx = cl.argmax(-1)
        el, iou, pc = self._edges_at(h, v, q_img, centers, pred_idx[:, None])
        return dict(presence_logit=pres, center_logits=cl, pred_idx=pred_idx, obj_logits=ol.masked_fill(~valid, -1e4),
                    box_pred=self.expected_box(el[:, 0], pc[:, 0]), iou_logit=iou[:, 0])

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
        el, _, pc = self._edges_at(h, v, q_img, centers, idx)
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
