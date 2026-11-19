"""Where does inference time go? Per-phase wall time (CUDA-synchronised) and the top CUDA kernels.

python scripts/profile_inference.py --ckpt CKPT --records REC --image_root IMG [--n 128] [--kernels]
Phases: preprocess (PIL + image processor, CPU) | collate | vision tower | modality stage (image + text streams)
        | fusion stage | heads | multi-box decode. Workloads: one image x N requests (image_first),
        one request x N images (text_first).
"""
import argparse
import collections
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glancex import GlanceX  # noqa: E402
from glancex.backbone import canonical_mode  # noqa: E402
from glancex.data import collate  # noqa: E402
from glancex.train import to_dev  # noqa: E402

T = collections.defaultdict(float)


def sync():
    torch.cuda.synchronize()


def wrap(obj, name, key):
    f = getattr(obj, name)

    def g(*a, **k):
        sync()
        t = time.perf_counter()
        out = f(*a, **k)
        sync()
        T[key] += time.perf_counter() - t
        return out
    setattr(obj, name, g)


@torch.inference_mode()
def run(m, items_fn, mode):
    T.clear()
    sync()
    t0 = time.perf_counter()
    t = time.perf_counter()
    items = items_fn()
    T["preprocess"] += time.perf_counter() - t
    t = time.perf_counter()
    b = collate(items, m.tok, m.cfg, canonical_mode(mode))
    b.pop("meta")
    b = to_dev(b, "cuda")
    sync()
    T["collate+h2d"] += time.perf_counter() - t
    from glancex.train import forward_batch
    out, _, (o, rv, rc) = forward_batch(m.net, m.heads, b, train=False)
    sync()
    t = time.perf_counter()
    m.heads.decode_multi(o["r"], o["vis"], rv, rc, o["vis_index"], out, thr=0.5)
    sync()
    T["decode_multi"] += time.perf_counter() - t
    T["TOTAL"] = time.perf_counter() - t0
    return {k: round(v * 1000, 1) for k, v in T.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--records", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--kernels", action="store_true")
    ap.add_argument("--exact", action="store_true", help="fast=False (unmerged LoRA, CPU preprocessing)")
    a = ap.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)
    m = GlanceX.from_pretrained(a.ckpt, base=a.base, device="cuda", fast=not a.exact)
    net, heads = m.net, m.heads
    wrap(net, "image_features", "vision_tower")
    wrap(net, "unimodal", "modality_stage")
    wrap(net, "fusion", "fusion_stage")
    wrap(heads, "forward", "heads")
    recs = [json.loads(line) for line in open(a.records)]
    ok = [r for r in recs if 280 <= r["W"] <= 1568 and 280 <= r["H"] <= 1568]
    main_rec = max(ok, key=lambda r: len(r["questions"]))
    texts = [q["text"] for q in main_rec["questions"]][: a.n]
    img = os.path.join(a.image_root, main_rec["file"])
    paths = [os.path.join(a.image_root, r["file"]) for r in ok if r is not main_rec][: a.n]
    work = {
        f"1 image x {a.n} requests (image_first)": (lambda: m._items([img], [texts]), "image_first"),
        f"1 request x {a.n} images (text_first)": (lambda: m._items(paths, [[texts[0]]] * len(paths)), "text_first"),
        "1 image x 1 request (image_first)": (lambda: m._items([img], [texts[:1]]), "image_first"),
    }
    for name, (fn, mode) in work.items():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            run(m, fn, mode)
        res = [run(m, fn, mode) for _ in range(3)]
        med = {k: sorted(r[k] for r in res)[1] for k in res[0]}
        print(f"\n== {name}")
        for k, v in med.items():
            print(f"  {k:16s} {v:9.1f} ms")
        print(f"  peak memory      {torch.cuda.max_memory_allocated() / 2**30:9.1f} GB")
        if a.kernels:
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                run(m, fn, mode)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18, max_name_column_width=70))


if __name__ == "__main__":
    main()
