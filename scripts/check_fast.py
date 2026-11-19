"""fast=True vs fast=False on the demo scenes: largest difference in P(present), confidence and box (pixels)."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glancex import GlanceX  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--base", default=None)
ap.add_argument("--spec", default="demo/scenes.json")
ap.add_argument("--image_root", required=True)
a = ap.parse_args()
scenes = json.load(open(a.spec))["scenes"]
p = lambda f: os.path.join(a.image_root, f)  # noqa: E731


def run(m):
    out = []
    for sc in scenes:
        if sc["kind"] == "search":
            out += m.search(sc["request"], [p(f) for f in sc["images"]], mode=sc["mode"], sort=False)
        else:
            out += m.ask(p(sc["image"]), sc["requests"], mode=sc["mode"])
    return out


ref = run(GlanceX.from_pretrained(a.ckpt, base=a.base, fast=False))
fast = run(GlanceX.from_pretrained(a.ckpt, base=a.base, fast=True))
# the saved demo predictions come from the original (pre-optimisation) code: the exact path must reproduce them
if os.path.exists("demo/predictions.json"):
    old = [r for sc in json.load(open("demo/predictions.json")) for r in sc["results"]]
    d0 = max(max(abs(o["present_prob"] - x.present_prob), *[abs(u - v) for u, v in zip(o["box"], x.box)])
             for o, x in zip(old, ref))
    print(f"exact path vs saved original predictions: max abs diff {d0:.2e}")
dp = max(abs(x.present_prob - y.present_prob) for x, y in zip(ref, fast))
dc = max(abs(x.confidence - y.confidence) for x, y in zip(ref, fast))
db = max(max(abs(u - v) for u, v in zip(x.box, y.box)) for x, y in zip(ref, fast))
nb = sum(len(x.boxes) != len(y.boxes) for x, y in zip(ref, fast))
flip = sum(x.present != y.present for x, y in zip(ref, fast))
print(f"{len(ref)} results: max |dP(present)| {dp:.4f}  max |dconf| {dc:.4f}  max |dbox| {db:.1f} px  "
      f"present flips {flip}  multi-box count changes {nb}")
