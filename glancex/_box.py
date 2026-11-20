"""Box primitive heads: presence gate + center-patch choice + 4 ordinal edge distributions (+ IoU-confidence head).

All geometry is in normalized image coordinates [0,1] (resize-invariant). Visual token (i,j) of an image with
grid (gh, gw) has center ((j+.5)/gw, (i+.5)/gh).
Edge offsets for a chosen patch center (px,py):  left=px-x1, top=py-y1, right=x2-px, bottom=y2-py, each a
distribution over K bins spanning [edge_lo, edge_hi].
"""
import math
import torch
import torch.nn as nn


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def fourier(xy, n=16):
    freqs = (2.0 ** torch.arange(n, device=xy.device, dtype=xy.dtype)) * math.pi
    a = xy[..., None] * freqs  # [...,2,n]
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)  # [..., 4n]


def box_iou_giou(a, b):
    """a,b: [...,4] xyxy. returns iou, giou."""
    x1 = torch.maximum(a[..., 0], b[..., 0]); y1 = torch.maximum(a[..., 1], b[..., 1])
    x2 = torch.minimum(a[..., 2], b[..., 2]); y2 = torch.minimum(a[..., 3], b[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area = lambda t: (t[..., 2] - t[..., 0]).clamp(min=0) * (t[..., 3] - t[..., 1]).clamp(min=0)
    union = area(a) + area(b) - inter
    iou = inter / union.clamp(min=1e-9)
    cx1 = torch.minimum(a[..., 0], b[..., 0]); cy1 = torch.minimum(a[..., 1], b[..., 1])
    cx2 = torch.maximum(a[..., 2], b[..., 2]); cy2 = torch.maximum(a[..., 3], b[..., 3])
    c = ((cx2 - cx1) * (cy2 - cy1)).clamp(min=1e-9)
    return iou, iou - (c - union) / c


class BoxHeads(nn.Module):
    def __init__(self, D, cfg):
        super().__init__()
        self.cfg = cfg
        d, K = cfg.head_dim, cfg.edge_bins
        self.K = K
        self.q_norm = nn.LayerNorm(D)
        self.v_norm = nn.LayerNorm(D)
        self.presence = mlp(D, d, 1)
        self.c_q = nn.Linear(D, d)
        self.c_k = nn.Linear(D, d)
        self.edge = mlp(2 * D + 64, 2 * d, 4 * K)
        self.iou = mlp(2 * D + 64, d, 1)
        self.register_buffer("bin_centers", torch.linspace(cfg.edge_lo, cfg.edge_hi, K), persistent=False)

    # ---------------------------------------------------------------- utils
    def patch_centers(self, gh, gw, device):
        ii, jj = torch.meshgrid(torch.arange(gh, device=device), torch.arange(gw, device=device), indexing="ij")
        return torch.stack([(jj.flatten() + 0.5) / gw, (ii.flatten() + 0.5) / gh], -1)  # [gh*gw, 2] (x,y)

    def _edge_feats(self, hq, v_sel, pc):
        return torch.cat([hq, v_sel, fourier(pc)], -1)

    def edges_logits(self, hq, v_sel, pc):
        return self.edge(self._edge_feats(hq, v_sel, pc)).view(-1, 4, self.K)

    def expected_box(self, edge_logits, pc):
        p = edge_logits.softmax(-1)
        e = (p * self.bin_centers).sum(-1)  # [M,4]  l,t,r,b
        return torch.stack([pc[:, 0] - e[:, 0], pc[:, 1] - e[:, 1], pc[:, 0] + e[:, 2], pc[:, 1] + e[:, 3]], -1)

    # ---------------------------------------------------------------- forward
    def forward(self, hq, vis, vis_valid, centers, q_img, sel_idx=None):
        """hq:[M,D] question readout; vis:[B,N,D] visual tokens per image (padded); vis_valid:[B,N];
        centers:[B,N,2] patch centers; q_img:[M] image index of each question.
        sel_idx:[M] patch to condition edges on (training: sampled from target). Heads run in fp32."""
        hq = self.q_norm(hq.float())
        vis = self.v_norm(vis.float())
        pres = self.presence(hq).squeeze(-1)
        kv = self.c_k(vis)                                    # [B,N,d]
        cl = torch.einsum("md,mnd->mn", self.c_q(hq), kv[q_img]) / math.sqrt(self.c_q.out_features)
        valid = vis_valid[q_img]
        cl = cl.masked_fill(~valid, -1e4)
        pred_idx = cl.argmax(-1)
        out = dict(presence_logit=pres, center_logits=cl, pred_idx=pred_idx)
        v_pred, pc_pred = vis[q_img, pred_idx], centers[q_img, pred_idx]
        el_pred = self.edges_logits(hq, v_pred, pc_pred)
        out["edge_logits_pred"] = el_pred
        out["box_pred"] = self.expected_box(el_pred, pc_pred)
        out["iou_logit"] = self.iou(self._edge_feats(hq, v_pred, pc_pred)).squeeze(-1)
        if sel_idx is not None:
            v_sel, pc_sel = vis[q_img, sel_idx], centers[q_img, sel_idx]
            el = self.edges_logits(hq, v_sel, pc_sel)
            out.update(edge_logits=el, pc_sel=pc_sel, box_sel=self.expected_box(el, pc_sel))
        return out


# ---------------------------------------------------------------- targets / losses
