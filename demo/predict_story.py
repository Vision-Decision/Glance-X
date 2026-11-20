"""Predictions for the demo story (demo/render_story.py): one request searched over 256 cached images, then new
requests on the image it picks. Everything in the video comes from this file's output.

python demo/predict_story.py --ckpt Vision-Decision/Glance-X-Lite --records RECORDS.jsonl --image_root IMG --out demo/story_predictions.json
"""
import argparse
import dataclasses
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glancex import GlanceX  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="Vision-Decision/Glance-X-Lite")
ap.add_argument("--base", default=None)
ap.add_argument("--spec", default="demo/story.json")
ap.add_argument("--records", required=True)
ap.add_argument("--image_root", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
spec = json.load(open(a.spec))
recs = [json.loads(line) for line in open(a.records)][: spec["n_images"]]
files = [r["file"] for r in recs]
paths = [os.path.join(a.image_root, f) for f in files]
m = GlanceX.from_pretrained(a.ckpt, base=a.base)


def timed(fn, n=5):
    fn()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts), out


t = time.perf_counter()
bank = m.encode(paths)
torch.cuda.synchronize()
encode_ms = (time.perf_counter() - t) * 1000
search_ms, ranked = timed(lambda: m.search(spec["search"], bank, sort=False))
best = max(ranked, key=lambda r: r.confidence)
pick = best.image_index
zoom_ms, zoom = timed(lambda: m.ask(paths[pick], spec["zoom_requests"]))
out = dict(gpu=torch.cuda.get_device_name(), files=files, search=spec["search"], encode_ms=encode_ms,
           search_ms=search_ms, ranking=[dataclasses.asdict(r) | dict(confidence=r.confidence) for r in ranked],
           pick=pick, zoom_file=files[pick], zoom_ms=zoom_ms,
           zoom=[dataclasses.asdict(r) | dict(confidence=r.confidence) for r in zoom])
json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
top = sorted(ranked, key=lambda r: -r.confidence)[:5]
print(f"encode {len(paths)} images {encode_ms:.0f} ms | search {search_ms:.1f} ms | pick {files[pick]} "
      f"| top5 {[(r.image_index, round(r.confidence, 3)) for r in top]} | zoom {zoom_ms:.1f} ms")
for r in zoom:
    print(" ", repr(r))
