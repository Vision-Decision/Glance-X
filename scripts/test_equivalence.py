"""Exactness tests for the Glance-X backbone (glancex/backbone.py) on a tiny random Qwen3.5 config (GPU, fp32).

1. K=0 + causal block mask: each direction equals a plain HF causal forward of the equivalent single sequence
   with the same M-RoPE positions:   i2t = [image, text, R]   t2i = [text, image, R]   bi = [image, text, R]
   (checks the readout R and every image token).
2. K=N (all layers unimodal): the text readout does not depend on the image.
3. Isolation: changing question 1 never changes question 0, in every direction.
4. The unimodal text stream is invariant to a constant position shift (so cached text works for any image).

python scripts/test_equivalence.py --model_id Qwen/Qwen3.5-4B-Base    # needs a CUDA GPU (fla kernels)
"""
import argparse, json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from glancex.backbone import GlanceXBackbone
from glancex.config import Config
from glancex.data import RecordDataset, collate

ap = argparse.ArgumentParser()
ap.add_argument("--model_id", required=True)
a = ap.parse_args()
dev = "cuda"
torch.manual_seed(0)
proc = AutoProcessor.from_pretrained(a.model_id)
c = AutoConfig.from_pretrained(a.model_id)
t = c.text_config
t.hidden_size, t.intermediate_size, t.num_hidden_layers = 128, 256, 8
t.layer_types = t.layer_types[:8]
t.num_attention_heads, t.num_key_value_heads, t.head_dim = 4, 2, 32
t.linear_num_key_heads, t.linear_num_value_heads, t.linear_key_head_dim, t.linear_value_head_dim = 2, 4, 16, 16
v = c.vision_config
v.depth, v.hidden_size, v.intermediate_size, v.num_heads, v.out_hidden_size = 1, 64, 128, 2, 128
hf = AutoModelForImageTextToText.from_config(c, dtype=torch.float32).to(dev).eval()

tmp = tempfile.mkdtemp()
with open(f"{tmp}/r.jsonl", "w") as f:
    for i, (W, H) in enumerate([(320, 240), (200, 300)]):
        Image.fromarray((np.random.rand(H, W, 3) * 255).astype("uint8")).save(f"{tmp}/im{i}.jpg")
        f.write(json.dumps(dict(file=f"im{i}.jpg", W=W, H=H, questions=[
            dict(text="the red car on the left", present=True, box=[1, 2, 50, 60]),
            dict(text="a dog", present=False, box=None),
            dict(text="two people", present=True, boxes=[[10, 20, 100, 120], [120, 20, 180, 140]])])) + "\n")

errs, ok = {}, True


def make(K, mask="causal"):
    cfg = Config().update(dict(bidir_k=K, bidir_block_mask=mask, max_vis_tokens=64, grad_ckpt=False,
                               pad_img_to=0, pad_txt_to=0, pad_txt_block_to=0, pad_t2i_block_to=0, pad_bi_block_to=0))
    net = GlanceXBackbone(hf, cfg).eval()
    ds = RecordDataset(f"{tmp}/r.jsonl", tmp, proc, cfg, train=False)
    return cfg, net, ds


def batch_of(ds, cfg, mode):
    b = collate([ds[0], ds[1]], ds.tok, cfg, mode); b.pop("meta")
    return {k: (x.to(dev) if torch.is_tensor(x) else x) for k, x in b.items()}


@torch.no_grad()
def reference(net, b, feats, m, order):
    """HF causal forward of the single sequence for question m (order: list of 'img' / 'txt' / 'txt_noR' / 'R')."""
    emb = net.lm.embed_tokens
    bi = int(b["q_img"][m])
    xi = emb(b["img_ids"][bi:bi + 1]).masked_scatter(b["img_mask"][bi:bi + 1][..., None], feats[bi][None])
    xt = emb(b["txt_ids"][m:m + 1])
    iv, tv = b["img_valid"][bi], b["txt_valid"][m]
    parts = {"img": (xi[0][iv], b["img_pos"][:, bi][:, iv], b["img_mask"][bi][iv]),
             "txt": (xt[0][tv], b["txt_pos"][:, m][:, tv], torch.zeros(int(tv.sum()), dtype=torch.bool, device=dev))}
    tx, tp, tf = parts["txt"]                       # x [L,D], pos [3,L], flag [L]
    parts["txt_noR"] = (tx[:-1], tp[:, :-1], tf[:-1])
    parts["R"] = (tx[-1:], tp[:, -1:], tf[-1:])
    xs, ps, fl = zip(*[parts[o] for o in order])
    x, pos, flag = torch.cat(xs)[None], torch.cat(ps, 1)[:, None], torch.cat(fl)
    h = net.lm(inputs_embeds=x, position_ids=pos).last_hidden_state[0]
    return h[-1], h[flag]


# 1. equivalence with HF, K=0, causal block mask
cfg, net, ds = make(0)
orders = {"i2t": ["img", "txt"], "t2i": ["txt_noR", "img", "R"], "bi": ["img", "txt"]}
for mode, order in orders.items():
    b = batch_of(ds, cfg, mode)
    with torch.no_grad():
        feats = net.image_features(b["pixel_values"], b["grid_thw"])
        o = net(b, feats, mode)
    for m in range(o["r"].shape[0]):
        r_ref, vis_ref = reference(net, b, feats, m, order)
        e_r = ((o["r"][m] - r_ref).abs().max() / r_ref.abs().max()).item()
        row = int(o["vis_index"][m])
        vis = o["vis"][row][b["vis_valid"][b["q_img"][m]]]
        e_v = ((vis - vis_ref).abs().max() / vis_ref.abs().max()).item()
        errs[f"{mode}_q{m}"] = max(e_r, e_v)
ok &= all(v < 5e-4 for v in errs.values())

# 2. K = N: text readout independent of the image
cfg, net, ds = make(len(hf.config.text_config.layer_types))
b = batch_of(ds, cfg, "i2t")
with torch.no_grad():
    feats = net.image_features(b["pixel_values"], b["grid_thw"])
    r1 = net(b, feats, "i2t")["r"]
    r2 = net(b, [f + 1.0 for f in feats], "i2t")["r"]
errs["K=N_text_independent_of_image"] = (r1 - r2).abs().max().item()
ok &= errs["K=N_text_independent_of_image"] == 0

# 3. isolation between questions, K=4, real (bidirectional) block mask
cfg, net, ds = make(4, "block_bidir")
for mode in orders:
    b = batch_of(ds, cfg, mode)
    with torch.no_grad():
        feats = net.image_features(b["pixel_values"], b["grid_thw"])
        r1 = net(b, feats, mode)["r"]
        b2 = dict(b); ti = b["txt_ids"].clone(); ti[1, -3] = ti[1, -3] + 5; b2["txt_ids"] = ti
        r2 = net(b2, feats, mode)["r"]
    errs[f"iso_{mode}_q0"] = (r2[0] - r1[0]).abs().max().item()
    errs[f"iso_{mode}_q1_changed"] = (r2[1] - r1[1]).abs().max().item()
    ok &= errs[f"iso_{mode}_q0"] == 0 and errs[f"iso_{mode}_q1_changed"] > 0

# 4. unimodal text stream invariant to a position shift
layers = list(zip(net.lm.layers, net.layer_types))[:4]
with torch.no_grad():
    xt = net.lm.embed_tokens(b["txt_ids"])
    h1 = net.unimodal(xt, b["txt_valid"], b["txt_pos"], layers)
    h2 = net.unimodal(xt, b["txt_valid"], b["txt_pos"] + 37, layers)
errs["text_shift_invariance"] = ((h1 - h2)[b["txt_valid"]].abs().max() / h1.abs().max()).item()
ok &= errs["text_shift_invariance"] < 1e-4

# 5. image cache: query_bank(encode_images(batch)) == the image_first forward (readout and image tokens)
for K in (0, 4):
    cfg, net, ds = make(K, "block_bidir")
    b = batch_of(ds, cfg, "i2t")
    with torch.no_grad():
        feats = net.image_features(b["pixel_values"], b["grid_thw"])
        ref = net(b, feats, "i2t")
        bank = net.encode_images(b, feats)
        o = net.query_bank(bank, b, b["q_img"])
    errs[f"bank_K{K}_readout"] = ((o["r"] - ref["r"]).abs().max() / ref["r"].abs().max()).item()
    errs[f"bank_K{K}_image_tokens"] = ((o["vis"] - ref["vis"]).abs().max() / ref["vis"].abs().max()).item()
    # one request shared by every image (q_img=None): request 0 of image 0 asked against image 0 and image 1
    one = dict(txt_ids=b["txt_ids"][:1], txt_valid=b["txt_valid"][:1], txt_pos=b["txt_pos"][:, :1])
    with torch.no_grad():
        o1 = net.query_bank(bank, one)
        b2 = dict(b, txt_ids=b["txt_ids"][:1].expand(2, -1), txt_valid=b["txt_valid"][:1].expand(2, -1),
                  txt_pos=b["txt_pos"][:, :1].expand(-1, 2, -1), q_img=torch.tensor([0, 1], device=dev))
        r2 = net(b2, feats, "i2t")["r"]
    errs[f"bank_K{K}_one_request_all_images"] = ((o1["r"] - r2).abs().max() / r2.abs().max()).item()
    ok &= all(errs[f"bank_K{K}_{n}"] < 1e-5 for n in ("readout", "image_tokens", "one_request_all_images"))

for k, x in errs.items():
    print(f"{k:34s} {x:.3e}")
print("ALL_OK" if ok else "FAIL")
