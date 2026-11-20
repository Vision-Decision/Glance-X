"""Evaluate a Glance-X checkpoint in one direction.

python -m glancex.evaluate --ckpt Vision-Decision/Glance-X-Lite --mode image_first|text_first \
    --sets refcoco_val,refcocog_val,grefcoco_val_merged \
    [--glance_data GLANCE_DATA_VAL.jsonl] [--limit_images 300] --out res.json

Two groups of results: the original benchmarks (RefCOCO / RefCOCOg / gRefCOCO, human labels) and our own
Glance-Data eval set (pipeline-generated, held-out images), reported per kind / language. Per set:
  acc50          single-target questions: argmax-center box IoU >= 0.5 with the target
  multi_f1       multi-target questions: decoded box set vs targets (greedy match at IoU 0.5)
  multi_hit_any  multi-target questions: the single box hits one of the targets
  reject_rate    absent questions with P(present) < 0.5;   present_rate on positives;  auroc of P(present)
"""
import argparse, collections, json, time
import numpy as np
import torch

from ._box import box_iou_giou
from .backbone import canonical_mode
from .build import build, load_ckpt, load_state
from .config import Config
from .data import RecordDataset, collate
from .forward import forward_batch, to_dev


def auroc(p, y):
    """Area under the ROC curve of scores p for binary labels y (ties get the average rank)."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv)
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / cnt)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def iou1(a, b):
    return float(box_iou_giou(torch.tensor(a)[None], torch.tensor(b)[None])[0][0])


def f1_sets(pred, gt, thr=0.5):
    used, tp = set(), 0
    for p in pred:
        best, bj = 0.0, -1
        for j, g in enumerate(gt):
            if j not in used:
                v = iou1(p, g)
                if v > best:
                    best, bj = v, j
        if best >= thr:
            used.add(bj); tp += 1
    prec = tp / max(1, len(pred)); rec = tp / max(1, len(gt))
    return 0.0 if tp == 0 else 2 * prec * rec / (prec + rec)


@torch.no_grad()
def predict(net, heads, ds, mode, images_per_batch, limit_images, obj_thr):
    n = len(ds) if not limit_images else min(limit_images, len(ds))
    tok, cfg = ds.tok, net.cfg
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(n)), batch_size=images_per_batch, num_workers=6,
                                     collate_fn=lambda it: collate(it, tok, cfg, mode))
    rows = []
    for batch in dl:
        meta = batch.pop("meta")
        b = to_dev(batch, "cuda")
        out, _, (o, rv, rc) = forward_batch(net, heads, b, train=False)
        multi = heads.decode_multi(o["r"], o["vis"], rv, rc, o["vis_index"], out, thr=obj_thr)
        p = out["presence_logit"].sigmoid().cpu()
        piou = out["iou_logit"].sigmoid().cpu()
        box = out["box_pred"].clamp(0, 1).cpu()
        for k, mt in enumerate(meta):
            q = mt["q"]
            nb = int(b["n_boxes"][k])
            rows.append(dict(kind=q.get("kind", "orig"), lang=q.get("lang", "en"), present=bool(b["present"][k]),
                             n_targets=nb, p_present=float(p[k]), p_iou=float(piou[k]), box=box[k].tolist(),
                             multi=[x.tolist() for x in multi[k][0].cpu()], gt=b["boxes"][k, :nb].cpu().tolist()))
    return rows


def summarize(rows):
    s = dict(n=len(rows))
    single = [r for r in rows if r["present"] and r["n_targets"] == 1]
    multi = [r for r in rows if r["present"] and r["n_targets"] > 1]
    absent = [r for r in rows if not r["present"]]
    pos = single + multi
    if single:
        s["acc50"] = float(np.mean([iou1(r["box"], r["gt"][0]) >= 0.5 for r in single]))
    if multi:
        s["multi_f1"] = float(np.mean([f1_sets(r["multi"], r["gt"]) for r in multi]))
        s["multi_hit_any"] = float(np.mean([max(iou1(r["box"], g) for g in r["gt"]) >= 0.5 for r in multi]))
        s["multi_count_acc"] = float(np.mean([len(r["multi"]) == r["n_targets"] for r in multi]))
    if pos:
        s["present_rate"] = float(np.mean([r["p_present"] >= 0.5 for r in pos]))
    if absent:
        s["reject_rate"] = float(np.mean([r["p_present"] < 0.5 for r in absent]))
    if pos and absent:
        s["auroc"] = auroc([r["p_present"] for r in rows], [int(r["present"]) for r in rows])
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", required=True, help="image_first | text_first | joint")
    ap.add_argument("--sets", default="refcoco_val,refcocog_val,grefcoco_val_merged")
    ap.add_argument("--prep_dir", default="data/prep", help="folder with eval_<set>.jsonl records")
    ap.add_argument("--image_root", default="data/images/coco_train2014", help="COCO train2014 images")
    ap.add_argument("--glance_data", "--diag", dest="glance_data", default="",
                    help="Glance-Data eval set (held-out images, pipeline-generated)")
    ap.add_argument("--limit_images", type=int, default=0)
    ap.add_argument("--images_per_batch", type=int, default=0, help="0: 8 for i2t, 1 for t2i / bi")
    ap.add_argument("--obj_thr", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.mode = canonical_mode(a.mode)
    torch.backends.cuda.enable_cudnn_sdp(False)
    sd = load_state(a.ckpt)                                   # Hugging Face id, exported folder or training .pt
    cfg = Config().update(sd["cfg"]).update(dict(pad_img_to=0, pad_txt_to=0, pad_txt_block_to=0, pad_t2i_block_to=0,
                                                 pad_bi_block_to=0))
    net, heads, proc = build(cfg, "cuda")
    load_ckpt(sd, net, heads)
    net.eval(); heads.eval()
    ipb = a.images_per_batch or (8 if a.mode == "i2t" else 1)
    res = dict(ckpt=a.ckpt, step=sd.get("extra", {}).get("step"), mode=a.mode, sets={})  # sets: originals + "glance_data"
    files = [(n, f"{a.prep_dir}/eval_{n}.jsonl") for n in a.sets.split(",") if n] + ([("glance_data", a.glance_data)] if a.glance_data else [])
    for name, path in files:
        ds = RecordDataset(path, a.image_root, proc, cfg, train=False)
        t0 = time.time()
        rows = predict(net, heads, ds, a.mode, ipb, a.limit_images, a.obj_thr)
        s = summarize(rows)
        s["wall_s"] = time.time() - t0
        if name == "glance_data":
            by = collections.defaultdict(list)
            for r in rows:
                by[r["kind"]].append(r); by[f"{r['kind']}:{r['lang']}"].append(r)
            s["by_kind"] = {k: summarize(v) for k, v in sorted(by.items())}
        res["sets"][name] = s
        print(name, json.dumps({k: v for k, v in s.items() if k != "by_kind"}), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
