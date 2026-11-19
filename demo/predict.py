"""Run Glance-X on a demo spec and dump every prediction to JSON (input of demo/render.py).

python demo/predict.py --ckpt glance-x-4b-k8.pt --spec demo/scenes.json --image_root data/images --out demo/predictions.json

spec: {"scenes": [
  {"kind": "ask",    "mode": "image_first", "image": "a.jpg", "requests": ["...", "..."]},
  {"kind": "find",   "mode": "joint",       "image": "b.jpg", "requests": ["..."]},
  {"kind": "search", "mode": "text_first",  "request": "...", "images": ["c.jpg", "d.jpg"]}]}
"""
import argparse
import dataclasses
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glancex import GlanceX  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--base", default=None, help="Qwen3.5 base id or local path (default: the one in the checkpoint)")
ap.add_argument("--spec", required=True)
ap.add_argument("--image_root", default=".")
ap.add_argument("--out", required=True)
a = ap.parse_args()

m = GlanceX.from_pretrained(a.ckpt, base=a.base)
spec = json.load(open(a.spec))
p = lambda f: os.path.join(a.image_root, f)  # noqa: E731
out = []
for sc in spec["scenes"]:
    t = time.time()
    if sc["kind"] == "search":
        res = m.search(sc["request"], [p(f) for f in sc["images"]], mode=sc.get("mode", "text_first"), sort=False)
    else:
        res = m.ask(p(sc["image"]), sc["requests"], mode=sc.get("mode", "image_first"))
    ms = (time.time() - t) * 1000
    out.append(dict(sc, results=[dict(dataclasses.asdict(r), confidence=r.confidence) for r in res], ms=ms))
    print(f"{sc['kind']:6s} {sc.get('mode', ''):11s} {len(res):3d} results  {ms:.0f} ms", flush=True)
json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
print("->", a.out)
