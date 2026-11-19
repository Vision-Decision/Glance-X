"""Build the original-data records (RefCOCO / RefCOCO+ / RefCOCOg / gRefCOCO) used for training and evaluation.

python scripts/prepare_refcoco.py --ann data/ann --out data/prep

Inputs (see docs/TRAINING.md):
  data/ann/{refcoco,refcocoplus,refcocog}/{train,validation,test,testB}.parquet  (HF jxu124/*)
  data/ann/grefcoco/grefs_unc.json                                                (HF FudanCVL/gRefCOCO)
  data/ann/annotations/instances_train2014.json                                   (COCO 2014)

Outputs (data/prep/):
  train.jsonl          one line per image-record: shared image + list of question blocks
  eval_<name>.jsonl    one line per image: questions of that eval split
  stats.json

Decisions:
  * gRefCOCO train multi-target expressions are not used here (Glance-X gets its multi-target training data from
    Glance-Data); single-target ones are identical to RefCOCO train and are de-duplicated; no-target ones become
    absent questions.
  * Extra negatives: expressions from other images whose referent COCO category has zero
    instances in this image (checked with COCO instances). Only "descriptive" expressions are
    swapped (RefCOCOg, or >=3 words from RefCOCO/+) to avoid location-only phrases like "left".
    Target overall absent ratio ~25%.
  * Every image used by ANY eval split (RefCOCO/+/g val/test*, gRefCOCO val/testA/testB) is
    removed from training (RefCOCOg-umd train overlaps RefCOCO unc val/test images).
"""
import argparse, json, os, random, re, collections
import pandas as pd

def norm_text(s):
    return re.sub(r"\s+", " ", s.strip()).rstrip(".").strip()

def load_parquet_refs(path, src):
    df = pd.read_parquet(path)
    out = []
    for r in df.itertuples():
        info = json.loads(r.raw_image_info)
        box = [float(x) for x in r.bbox]  # xyxy
        for s in r.sentences:
            out.append(dict(image_id=int(r.image_id), file=info["file_name"], W=info["width"], H=info["height"],
                            text=norm_text(s["sent"]), raw=s["raw"], box=box, present=True, cat=int(r.category_id),
                            src=src, sent_id=int(s["sent_id"])))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="data/ann")
    ap.add_argument("--out", default="data/prep")
    ap.add_argument("--absent_ratio", type=float, default=0.25)
    ap.add_argument("--max_q_per_record", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)

    coco = json.load(open(f"{a.ann}/annotations/instances_train2014.json"))
    img_info = {im["id"]: im for im in coco["images"]}
    img_cats = collections.defaultdict(set)
    ann_box = {}
    for an in coco["annotations"]:
        img_cats[an["image_id"]].add(an["category_id"])
        x, y, w, h = an["bbox"]
        ann_box[an["id"]] = (an["image_id"], an["category_id"], [x, y, x + w, y + h])

    split_files = {"train": "train", "val": "validation", "testA": "test", "testB": "testB", "test": "test"}
    evals, train_pos = {}, []
    for ds in ["refcoco", "refcocoplus", "refcocog"]:
        for sp in (["train", "val", "testA", "testB"] if ds != "refcocog" else ["train", "val", "test"]):
            items = load_parquet_refs(f"{a.ann}/{ds}/{split_files[sp]}.parquet", ds)
            if sp == "train":
                train_pos += items
            else:
                evals[f"{ds}_{sp}"] = items

    # gRefCOCO
    grefs = json.load(open(f"{a.ann}/grefcoco/grefs_unc.json"))
    g_train_neg, n_multi = [], 0
    for sp in ["val", "testA", "testB"]:
        evals[f"grefcoco_{sp}"] = []
    for r in grefs:
        iid = r["image_id"]; im = img_info[iid]
        ann_ids = [x for x in r["ann_id"] if x != -1] if isinstance(r["ann_id"], list) else [r["ann_id"]]
        no_t = bool(r["no_target"])
        boxes = [ann_box[x][2] for x in ann_ids]
        for s in r["sentences"]:
            q = dict(image_id=iid, file=im["file_name"], W=im["width"], H=im["height"], text=norm_text(s["sent"]),
                     raw=s["raw"], src="grefcoco", sent_id=int(s["sent_id"]), present=not no_t,
                     box=boxes[0] if len(boxes) == 1 else None, n_targets=len(boxes),
                     boxes=boxes, cat=(ann_box[ann_ids[0]][1] if ann_ids else -1))
            if r["split"] == "train":
                if no_t:
                    g_train_neg.append(q)
                elif len(boxes) > 1:
                    n_multi += 1  # not used (see docstring)
                # single-target == RefCOCO train (dedup below)
            else:
                evals[f"grefcoco_{r['split']}"].append(q)

    # this gRefCOCO release has no single-target val expressions (they are the RefCOCO-unc val refs), so build a
    # merged set: gRefCOCO val (no-target + multi-target) + RefCOCO val (single-target), same images.
    evals["grefcoco_val_merged"] = evals["grefcoco_val"] + [dict(q, n_targets=1, boxes=[q["box"]]) for q in evals["refcoco_val"]]
    held_out_imgs = set()
    for v in evals.values():
        held_out_imgs |= {q["image_id"] for q in v}
    n_before = len(train_pos)
    train_pos = [q for q in train_pos if q["image_id"] not in held_out_imgs]
    g_train_neg = [q for q in g_train_neg if q["image_id"] not in held_out_imgs]
    print(f"train positives {n_before} -> {len(train_pos)} after removing {len(held_out_imgs)} held-out images")

    # dedupe (image, text)
    def dedupe(qs):
        seen, out = set(), []
        for q in qs:
            k = (q["image_id"], q["text"].lower())
            if k in seen: continue
            seen.add(k); out.append(q)
        return out
    train_pos = dedupe(train_pos)
    pos_keys = {(q["image_id"], q["text"].lower()) for q in train_pos}
    g_train_neg = [q for q in dedupe(g_train_neg) if (q["image_id"], q["text"].lower()) not in pos_keys]

    # swapped negatives
    n_pos, n_gneg = len(train_pos), len(g_train_neg)
    n_total_target = int((n_pos) / (1 - a.absent_ratio))
    n_swap = max(0, n_total_target - n_pos - n_gneg)
    donors = [q for q in train_pos if q["src"] == "refcocog" or len(q["text"].split()) >= 3]
    imgs = sorted({q["image_id"] for q in train_pos})
    img_meta = {q["image_id"]: (q["file"], q["W"], q["H"]) for q in train_pos}
    swap_neg, tries = [], 0
    existing = set(pos_keys) | {(q["image_id"], q["text"].lower()) for q in g_train_neg}
    while len(swap_neg) < n_swap and tries < n_swap * 20:
        tries += 1
        iid = rng.choice(imgs); d = rng.choice(donors)
        if d["image_id"] == iid or d["cat"] in img_cats[iid]:
            continue
        k = (iid, d["text"].lower())
        if k in existing: continue
        existing.add(k)
        f, W, H = img_meta[iid]
        swap_neg.append(dict(image_id=iid, file=f, W=W, H=H, text=d["text"], src="swap", present=False, box=None,
                             cat=d["cat"], sent_id=-1))

    allq = train_pos + g_train_neg + swap_neg
    by_img = collections.defaultdict(list)
    for q in allq:
        by_img[q["image_id"]].append(q)
    n_rec = 0
    with open(f"{a.out}/train.jsonl", "w") as f:
        for iid in sorted(by_img):
            qs = by_img[iid]; rng.shuffle(qs)
            q0 = qs[0]
            for i in range(0, len(qs), a.max_q_per_record):
                chunk = qs[i:i + a.max_q_per_record]
                f.write(json.dumps(dict(image_id=iid, file=q0["file"], W=q0["W"], H=q0["H"],
                                        questions=[dict(text=q["text"], present=q["present"], box=q["box"], src=q["src"])
                                                   for q in chunk])) + "\n")
                n_rec += 1

    eval_stats = {}
    for name, qs in evals.items():
        g = collections.defaultdict(list)
        for q in qs: g[q["image_id"]].append(q)
        with open(f"{a.out}/eval_{name}.jsonl", "w") as f:
            for iid, lst in sorted(g.items()):
                f.write(json.dumps(dict(image_id=iid, file=lst[0]["file"], W=lst[0]["W"], H=lst[0]["H"],
                                        questions=[dict(text=q["text"], present=q["present"], box=q["box"],
                                                        n_targets=q.get("n_targets", 1), boxes=q.get("boxes"),
                                                        sent_id=q["sent_id"]) for q in lst])) + "\n")
        eval_stats[name] = dict(n_q=len(qs), n_img=len(g), n_absent=sum(not q["present"] for q in qs))
    stats = dict(train_images=len(by_img), train_records=n_rec, train_pos=n_pos, grefcoco_train_neg=n_gneg,
                 swap_neg=len(swap_neg), dropped_multi_target_sents=n_multi, held_out_images=len(held_out_imgs),
                 absent_ratio=(n_gneg + len(swap_neg)) / len(allq),
                 q_per_image_mean=len(allq) / len(by_img),
                 src_counts=collections.Counter(q["src"] for q in allq), eval=eval_stats)
    json.dump(stats, open(f"{a.out}/stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))

if __name__ == "__main__":
    main()
