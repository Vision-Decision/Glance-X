"""CPU unit tests: no model weights, no GPU."""
import math

import pytest
import torch

from glancex.backbone import canonical_mode, pack
from glancex.config import Config
from glancex.data import collate, resize_dims, targets_of


def test_mode_aliases():
    assert [canonical_mode(m) for m in ("image_first", "text_first", "joint")] == ["i2t", "t2i", "bi"]
    assert [canonical_mode(m) for m in ("i2t", "t2i", "bi")] == ["i2t", "t2i", "bi"]
    with pytest.raises(ValueError):
        canonical_mode("sideways")


@pytest.mark.parametrize("wh", [(640, 480), (480, 640), (1920, 1080), (100, 100), (2000, 300)])
def test_resize_dims(wh):
    w, h = resize_dims(*wh, max_tokens=640, max_upscale=2.0)
    assert w % 32 == 0 and h % 32 == 0
    assert (w // 32) * (h // 32) <= 640
    assert abs(math.log((w / h) / (wh[0] / wh[1]))) < 0.35        # aspect ratio roughly kept


def test_config_update():
    c = Config()
    assert c.bidir_k == 8 and c.model_id == "Qwen/Qwen3.5-4B-Base"
    d = c.update({"bidir_k": "0", "grad_ckpt": "false", "not_a_key": 1}, from_str=True)
    assert d.bidir_k == 0 and d.grad_ckpt is False and c.bidir_k == 8
    with pytest.raises(KeyError):
        c.update({"not_a_key": 1}, strict=True)


def test_pack_concatenates_valid_tokens():
    D = 2
    # row 0: segment A has 2 valid tokens, B has 1;  row 1: A has 1, B has 2   (inputs are LEFT padded)
    xa = torch.tensor([[[0, 0], [1, 1], [2, 2]], [[0, 0], [0, 0], [3, 3]]], dtype=torch.float)
    va = torch.tensor([[False, True, True], [False, False, True]])
    pa = torch.arange(3).expand(3, 2, 3) + 10
    xb = torch.tensor([[[0, 0], [7, 7]], [[8, 8], [9, 9]]], dtype=torch.float)
    vb = torch.tensor([[False, True], [True, True]])
    pb = torch.arange(2).expand(3, 2, 2) + 50
    fb = vb.clone()
    x, valid, pos, flags, last = pack([(xa, va, pa, None), (xb, vb, pb, fb)], T=5)
    assert x.shape == (2, 5, D)
    assert x[0, :3, 0].tolist() == [1, 2, 7] and x[1, :3, 0].tolist() == [3, 8, 9]
    assert valid.tolist() == [[True] * 3 + [False] * 2, [True] * 3 + [False] * 2]
    assert pos[0, 0, :3].tolist() == [11, 12, 51]
    assert pos[0, 0, 3:].tolist() == [52, 53]                     # pads continue after the last position
    assert flags[0].sum() == 0 and flags[1][0].tolist() == [False, False, True, False, False]
    assert last.tolist() == [2, 2]


class FakeTok:
    vs, ve, pad_img, pad = 101, 102, 103, 0


def _item(gh, gw, n_q):
    return dict(pixel_values=torch.zeros(gh * gw * 4, 8), grid_thw=torch.tensor([1, gh * 2, gw * 2]),
                qs=[dict(text=f"q{j}") for j in range(n_q)], q_ids=[[5] * (3 + j) for j in range(n_q)],
                present=[True] * n_q, boxes=[[[0.1, 0.1, 0.5, 0.5]]] * n_q, rec={}, orig_wh=(64, 64))


@pytest.mark.parametrize("mode", ["i2t", "t2i", "bi"])
def test_collate_layout(mode):
    cfg = Config().update(dict(pad_img_to=0, pad_txt_to=0))
    b = collate([_item(3, 4, 2), _item(2, 2, 1)], FakeTok(), cfg, mode)
    assert b["mode"] == mode and b["q_img"].tolist() == [0, 0, 1]
    assert b["img_ids"].shape == (2, 14)                          # 12 patches + vision_start/end, left padded
    assert b["img_ids"][1, :8].eq(FakeTok.pad).all() and b["img_ids"][1, 8] == FakeTok.vs
    assert int(b["img_mask"][0].sum()) == 12 and int(b["vis_valid"][1].sum()) == 4
    # text stream: left padded, positions start at the fixed text offset whatever the image
    for m, n in enumerate([3, 4, 3]):
        v = b["txt_valid"][m]
        assert int(v.sum()) == n and v[-n:].all()
        assert b["txt_pos"][0, m][v].tolist() == list(range(cfg.text_pos0, cfg.text_pos0 + n))
    # image positions: (1, 1+i, 1+j) on patches, vision_end at 1 + max(gh, gw)
    p = b["img_pos"][:, 0][:, b["img_mask"][0]]
    assert p[1].max() == 3 and p[2].max() == 4 and b["img_pos"][0, 0, -1] == 5
    assert torch.allclose(b["centers"][0, 0], torch.tensor([0.125, 1 / 6]))


def test_targets_of():
    assert targets_of(dict(present=True, box=[0, 0, 50, 25]), 100, 50, 16) == [[0, 0, 0.5, 0.5]]
    two = dict(present=True, box=None, boxes=[[0, 0, 10, 10], [50, 25, 100, 50]])
    assert len(targets_of(two, 100, 50, 16)) == 2
    assert targets_of(dict(present=False, box=[0, 0, 1, 1]), 100, 50, 16) == []
