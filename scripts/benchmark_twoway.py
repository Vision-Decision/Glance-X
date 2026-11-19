"""Latency of the two-way (image cache) design, same inputs as scripts/compare_jev.py.

python scripts/benchmark_twoway.py --ckpt CKPT --records REC --image_root IMG --out res.json [--devices ...]
  one image x N requests      : ask(image, N requests), image_first (+ two-way decoder)
  one request x N cached imgs : search(request, bank) with bank = encode(images) done beforehand
  one request x N new images  : encode(images) + search(request, bank), end to end
Wall clock, median of --repeat after --warmup, CUDA-synchronised.
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glancex import GlanceX  # noqa: E402
from scripts.compare_jev import acc50, pick  # noqa: E402


def timed(fn, warmup, repeat):
    out = None
    for _ in range(warmup):
        out = fn()
    ts = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--records", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--sizes", default="1,16,64,128")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)
    sizes = [int(x) for x in a.sizes.split(",")]
    sel = pick(a.records, a.image_root, max(sizes))
    m = GlanceX.from_pretrained(a.ckpt, base=a.base, device="cuda")
    res = dict(ckpt=a.ckpt, decoder_layers=m.cfg.decoder_layers, gpu=torch.cuda.get_device_name(),
               one_image_many_requests={}, one_request_cached_images={}, one_request_new_images={})
    for n in sizes:
        texts = [q["text"] for q in sel["questions"][:n]]
        ms, out = timed(lambda: m.ask(sel["image"], texts, mode="image_first"), a.warmup, a.repeat)
        res["one_image_many_requests"][n] = ms
        if n == max(sizes):
            res["acc50"], res["acc50_n"] = acc50(sel["questions"][:n], [r.box for r in out])
        print(f"1 image x {n:3d} requests        {ms:8.1f} ms", flush=True)
    for n in sizes:
        imgs = sel["search_images"][:n]
        bank = m.encode(imgs)
        ms, _ = timed(lambda: m.search(sel["search_request"], bank), a.warmup, a.repeat)
        res["one_request_cached_images"][n] = ms
        print(f"1 request x {n:3d} cached images  {ms:8.1f} ms", flush=True)
        ms2, _ = timed(lambda: m.search(sel["search_request"], m.encode(imgs)), a.warmup, a.repeat)
        res["one_request_new_images"][n] = ms2
        print(f"1 request x {n:3d} new images     {ms2:8.1f} ms (encode + search)", flush=True)
        bank = None   # free the cache before the next size
    b1 = m.encode(sel["search_images"][:8])
    res["bank_mb_per_image"] = b1.nbytes / 2**20 / len(b1)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}))


if __name__ == "__main__":
    main()
