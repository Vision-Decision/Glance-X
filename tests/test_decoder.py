"""The two-way decoder starts as the identity (zero-initialised residual heads) and still receives gradients."""
import torch

from glancex.config import Config
from glancex.heads import GlanceXHeads


def _inputs(M=5, R=2, N=12, T=6, D=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    hq = torch.randn(M, D, generator=g)
    vis = torch.randn(R, N, D, generator=g)
    vis_valid = torch.ones(R, N, dtype=torch.bool)
    vis_valid[1, 9:] = False
    centers = torch.rand(R, N, 2, generator=g)
    q_img = torch.tensor([0, 0, 1, 1, 1])
    q_tok = torch.randn(M, T, D, generator=g)
    q_valid = torch.ones(M, T, dtype=torch.bool)
    q_valid[0, 4:] = False
    r_idx = q_valid.sum(1) - 1
    return hq, vis, vis_valid, centers, q_img, q_tok, q_valid, r_idx


def _cfgs():
    base = Config().update(dict(head_dim=16, edge_bins=8))
    return base, base.update(dict(decoder_layers=2, decoder_dim=16, decoder_heads=2))


def test_decoder_starts_as_identity():
    c0, c1 = _cfgs()
    torch.manual_seed(0)
    h0 = GlanceXHeads(32, c0)
    h1 = GlanceXHeads(32, c1)
    missing, unexpected = h1.load_state_dict(h0.state_dict(), strict=False)
    assert not unexpected and missing
    hq, vis, vv, cen, qi, qt, qv, ri = _inputs()
    sel = torch.tensor([0, 1, 2, 3, 4])
    a = h0(hq, vis, vv, cen, qi, sel_idx=sel)
    b = h1(hq, vis, vv, cen, qi, sel_idx=sel, q_tok=qt, q_valid=qv, r_idx=ri)
    assert b["dec"] is not None
    for k in ("presence_logit", "center_logits", "obj_logits", "box_pred", "iou_logit", "edge_logits", "box_sel"):
        assert torch.allclose(a[k], b[k], atol=1e-6), k
    ma = h0.multi_tensors(hq, vis, vv, cen, qi, a)
    mb = h1.multi_tensors(hq, vis, vv, cen, qi, b)
    assert all(torch.allclose(x.float(), y.float(), atol=1e-6) for x, y in zip(ma, mb))


def test_decoder_gets_gradients():
    _, c1 = _cfgs()
    h1 = GlanceXHeads(32, c1)
    hq, vis, vv, cen, qi, qt, qv, ri = _inputs()
    out = h1(hq, vis, vv, cen, qi, sel_idx=torch.zeros(5, dtype=torch.long), q_tok=qt, q_valid=qv, r_idx=ri)
    loss = out["presence_logit"].sum() + out["center_logits"].logsumexp(-1).sum() + out["obj_logits"].sum() \
        + out["edge_logits"].sum() + out["iou_logit"].sum()
    loss.backward()
    for name in ("dc_k.weight", "do_k.weight", "dpres.2.weight", "dedge.2.weight", "diou.2.weight"):
        g = dict(h1.named_parameters())[name].grad
        assert g is not None and g.abs().sum() > 0, name
