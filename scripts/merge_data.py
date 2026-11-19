"""Merge the original grounding records and the Glance-Data augmentation into one record per image (Glance-X training).

python scripts/merge_data.py --orig data/orig/train.jsonl --aug data/glance-data/aug_train_all.jsonl --out data/train.jsonl
Original questions get orig=True and boxes=[box]; augmented ones keep their boxes / kind / lang.
Images that appear only in the augmentation are skipped (the augmentation is built on the training images).
"""
import argparse, collections, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--orig", required=True)
ap.add_argument("--aug", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
recs = {}
by_img = collections.defaultdict(list)
for line in open(a.orig):  # the original file may split one image into several records
    r = json.loads(line)
    recs.setdefault(r["image_id"], dict(image_id=r["image_id"], file=r["file"], W=r["W"], H=r["H"]))
    for q in r["questions"]:
        by_img[r["image_id"]].append(dict(q, orig=True, boxes=[q["box"]] if q.get("box") is not None else None,
                                          kind=q.get("src", "orig"), lang="en"))
n_aug = 0
for line in open(a.aug):
    r = json.loads(line)
    if r["image_id"] not in recs:
        continue
    for q in r["questions"]:
        by_img[r["image_id"]].append(dict(q, orig=False))
        n_aug += 1
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w") as f:
    for iid, r in recs.items():
        f.write(json.dumps(dict(r, questions=by_img[iid]), ensure_ascii=False) + "\n")
n = sum(len(v) for v in by_img.values())
print(f"images {len(recs)} questions {n} (orig {n - n_aug}, aug {n_aug}) -> {a.out}")
