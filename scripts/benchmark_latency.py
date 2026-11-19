"""Latency of the two workloads, per direction, at several batch sizes (CUDA).

python scripts/benchmark_latency.py --ckpt Vision-Decision/Glance-X-4B --records data/glance-data/aug_val.jsonl \
    --image_root data/images/coco_train2014 --sizes 1,16,64,128 --out results/latency.json

one image, N requests : ask(image, N requests)          image_first (and joint for comparison)
one request, N images : search(request, N images)       text_first  (and joint for comparison), all N in one batch
Times are the median of --repeat runs after --warmup runs (fla compiles kernels per shape on first use).
`model` = collate + forward + box decoding + copy to CPU, on preprocessed images.
`preprocess` = PIL resize + Qwen image processor, on CPU, per image.
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


def timed(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Vision-Decision/Glance-X-4B")
    ap.add_argument("--base", default=None)
    ap.add_argument("--records", required=True, help="record jsonl: images and their requests")
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--sizes", default="1,16,64,128")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)
    sizes = [int(x) for x in a.sizes.split(",")]
    N = max(sizes)
    m = GlanceX.from_pretrained(a.ckpt, base=a.base, device="cuda")
    recs = [json.loads(line) for line in open(a.records)]
    paths = [os.path.join(a.image_root, r["file"]) for r in recs[:N]]
    texts = [q["text"] for q in max(recs, key=lambda r: len(r["questions"]))["questions"]]
    assert len(paths) >= N and len(texts) >= N, "not enough images / requests in --records"
    request = texts[0]
    t = time.perf_counter()
    items1 = [m._item(p, [request]) for p in paths]
    prep_ms = (time.perf_counter() - t) * 1000 / len(paths)
    img_one = paths[0]
    res = dict(gpu=torch.cuda.get_device_name(), preprocess_ms_per_image=prep_ms, one_image_many_requests={},
               one_request_many_images={})
    for n in sizes:
        item = m._item(img_one, texts[:n])
        for mode in ("image_first", "joint"):
            ms = timed(lambda: m._run([item], mode, 0.5), a.warmup, a.repeat)
            res["one_image_many_requests"].setdefault(mode, {})[n] = ms
            print(f"1 image  x {n:3d} requests  {mode:11s} {ms:8.1f} ms", flush=True)
        for mode in ("text_first", "joint"):
            ms = timed(lambda: m._run(items1[:n], mode, 0.5), a.warmup, a.repeat)
            res["one_request_many_images"].setdefault(mode, {})[n] = ms
            print(f"1 request x {n:3d} images   {mode:11s} {ms:8.1f} ms", flush=True)
    res["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    print(f"preprocess {prep_ms:.1f} ms / image (CPU), peak memory {res['peak_mem_gb']:.1f} GB")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
