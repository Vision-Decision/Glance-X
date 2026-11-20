"""Speed (and box accuracy on the same samples) of Glance-X vs TypeSafe Jev's Box question.

Both systems get identical inputs, picked from a record file (default: the Glance-COCO eval set):
  one image, N requests : the record with the most requests; its first N requests
                          Jev       : ONE request, the image in `state`, N Box questions
                          Glance-X  : ask(image, N requests), image_first (one forward)
  one request, N images : the first N records' images (sides within Jev's 280..1568 px); one fixed request
                          Jev       : N concurrent requests, one image + one Box question each
                          Glance-X  : search(request, N images) in one batch, text_first

python scripts/compare_jev.py --system jev --key_file .secrets/typesafe_key --records REC --image_root IMG --out jev.json
python scripts/compare_jev.py --system glancex --ckpt Vision-Decision/Glance-X-4B --records REC --image_root IMG --out gx.json
Latency = wall clock of the whole call (Jev: HTTPS round trip from this machine; Glance-X: preprocessing + GPU).
"""
import argparse
import base64
import concurrent.futures as cf
import json
import os
import statistics
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

URL = "https://api.typesafe.ai/v1/systemone"


def pick(records, image_root, n_max):
    recs = [json.loads(line) for line in open(records)]
    ok = [r for r in recs if 280 <= r["W"] <= 1568 and 280 <= r["H"] <= 1568]
    main = max(ok, key=lambda r: len(r["questions"]))
    qs = main["questions"][:n_max]
    imgs = [r for r in ok if r is not main][:n_max]
    request = next(q["text"] for q in main["questions"] if q.get("kind") == "short" and q["present"])
    return dict(image=os.path.join(image_root, main["file"]), W=main["W"], H=main["H"], questions=qs,
                search_request=request, search_images=[os.path.join(image_root, r["file"]) for r in imgs])


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def acc50(qs, boxes):
    """Single-target present requests: predicted box (pixels) IoU >= 0.5 with the target."""
    hits = [iou(b, q["boxes"][0]) >= 0.5 for q, b in zip(qs, boxes) if q["present"] and len(q.get("boxes") or []) == 1]
    return (sum(hits) / len(hits), len(hits)) if hits else (None, 0)


# ------------------------------------------------------------------ Jev
class Jev:
    def __init__(self, key, model):
        import requests
        self.s = requests.Session()
        self.h = {"Authorization": f"Bearer {key}", "TypeSafe-Danger-Zone-Images": "true"}
        self.model = model
        self.version = None

    def call(self, image, instructions):
        """One request: `image` in the state, one Box question per instruction. Returns (boxes [l,t,r,b], conf)."""
        ref = uuid.uuid4().hex
        data = base64.b64encode(open(image, "rb").read()).decode()
        body = dict(model=self.model, state={"photo": ref},
                    assets={ref: {"type": "image", "format": "jpg", "data_base64": data}},
                    questions={f"q{i}": {"type": "box", "instructions": f"{t} in `photo`"}
                               for i, t in enumerate(instructions)})
        for attempt in range(6):
            r = self.s.post(URL, headers=self.h, json=body, timeout=300)
            if r.status_code in (429, 529):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            break
        else:
            r.raise_for_status()
        out = r.json()
        self.version = out.get("model")
        ans = [out["answers"][f"q{i}"] for i in range(len(instructions))]
        return [[a["box"]["left"], a["box"]["top"], a["box"]["right"], a["box"]["bottom"]] for a in ans], \
               [a.get("confidence") for a in ans]


def run_jev(a, sel, sizes):
    jev = Jev(open(a.key_file).read().strip(), a.model)
    res = dict(system="jev", one_image_many_requests={}, one_request_many_images={}, accuracy={})
    W, H = sel["W"], sel["H"]
    jev.call(sel["image"], [sel["questions"][0]["text"]])                 # warm-up
    for n in sizes:
        texts = [q["text"] for q in sel["questions"][:n]]
        ts, boxes = [], None
        for _ in range(a.repeat):
            t = time.perf_counter()
            b, _ = jev.call(sel["image"], texts)
            ts.append((time.perf_counter() - t) * 1000)
            boxes = boxes or b
        res["one_image_many_requests"][n] = dict(ms=statistics.median(ts), all_ms=ts)
        if n == max(sizes):
            px = [[x[0] * W, x[1] * H, x[2] * W, x[3] * H] for x in boxes]
            res["accuracy"]["acc50"], res["accuracy"]["n"] = acc50(sel["questions"][:n], px)
        print(f"jev  1 image x {n:3d} requests: {statistics.median(ts):8.0f} ms  {[round(x) for x in ts]}", flush=True)
    for n in sizes:
        imgs = sel["search_images"][:n]
        ts, per = [], []
        for _ in range(a.repeat):
            def one(p):
                t1 = time.perf_counter()
                jev.call(p, [sel["search_request"]])
                return (time.perf_counter() - t1) * 1000
            t = time.perf_counter()
            with cf.ThreadPoolExecutor(max_workers=min(n, a.concurrency)) as ex:
                per += list(ex.map(one, imgs))
            ts.append((time.perf_counter() - t) * 1000)
        res["one_request_many_images"][n] = dict(ms=statistics.median(ts), all_ms=ts,
                                                 per_request_median_ms=statistics.median(per))
        print(f"jev  1 request x {n:3d} images: {statistics.median(ts):8.0f} ms wall, "
              f"{statistics.median(per):.0f} ms per request (concurrency {min(n, a.concurrency)})", flush=True)
    res["model_version"] = jev.version
    return res


# ------------------------------------------------------------------ Glance-X
def run_glancex(a, sel, sizes):
    import torch

    from glancex import GlanceX
    torch.backends.cuda.enable_cudnn_sdp(False)
    devices = [d for d in a.devices.split(",") if d] or None
    m = GlanceX.from_pretrained(a.ckpt, base=a.base, device="cuda", devices=devices)
    res = dict(system="glancex", one_image_many_requests={}, one_request_many_images={}, accuracy={},
               gpu=torch.cuda.get_device_name(), n_gpus=len(m.replicas))

    def timed(fn):
        for _ in range(a.warmup):
            out = fn()
        ts = []
        for _ in range(a.repeat):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
        return statistics.median(ts), ts, out

    for n in sizes:
        texts = [q["text"] for q in sel["questions"][:n]]
        ms, ts, out = timed(lambda: m.ask(sel["image"], texts, mode="image_first"))
        res["one_image_many_requests"][n] = dict(ms=ms, all_ms=ts)
        if n == max(sizes):
            res["accuracy"]["acc50"], res["accuracy"]["n"] = acc50(sel["questions"][:n], [r.box for r in out])
        print(f"glancex 1 image x {n:3d} requests: {ms:8.1f} ms", flush=True)
    for n in sizes:
        imgs = sel["search_images"][:n]
        ms, ts, _ = timed(lambda: m.search(sel["search_request"], imgs, mode="text_first", batch_size=n))
        res["one_request_many_images"][n] = dict(ms=ms, all_ms=ts)
        print(f"glancex 1 request x {n:3d} images: {ms:8.1f} ms", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["jev", "glancex"], required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--sizes", default="1,16,64,128")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key_file", default="")
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--ckpt", default="Vision-Decision/Glance-X-4B")
    ap.add_argument("--base", default=None)
    ap.add_argument("--devices", default="", help="e.g. cuda:0,cuda:1,cuda:2 (Glance-X on several GPUs)")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]
    sel = pick(a.records, a.image_root, max(sizes))
    print(f"image {os.path.basename(sel['image'])} ({sel['W']}x{sel['H']}), {len(sel['questions'])} requests; "
          f"search {sel['search_request']!r} over {len(sel['search_images'])} images", flush=True)
    res = run_jev(a, sel, sizes) if a.system == "jev" else run_glancex(a, sel, sizes)
    res["inputs"] = dict(image=os.path.basename(sel["image"]), search_request=sel["search_request"],
                         search_images=[os.path.basename(p) for p in sel["search_images"]])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
