"""Save the fp32 MPS gate, or check the Mac implementation against it.

Run from the checkout with python -m scripts.check_mac. Outputs are local JSONL files.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from glancex import GlanceX
from glancex.data import collate, targets_of
from glancex.evaluate import summarize
from glancex.forward import to_dev
from glancex.inference import Result

MODES = ("image_first", "text_first")
CANDIDATE_MODES = (*MODES, "cached")
SETS = ("refcoco_val", "grefcoco_val")


def read_rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def gate_requests(root):
    by_file = defaultdict(list)
    for name in SETS:
        for rec in read_rows(root / f"eval_{name}.jsonl"):
            for i, q in enumerate(rec["questions"]):
                gt = targets_of(q, rec["W"], rec["H"], 16)
                by_file[rec["file"]].append(
                    dict(key=[name, rec["image_id"], i], request=q["text"], gt=gt, present=bool(gt),
                         wh=[rec["W"], rec["H"]])
                )
    missing = [name for name in by_file if not (root / "images" / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} gate images missing, starting with {missing[0]}")
    return by_file


@torch.no_grad()
def reference(model, it, b, feats, requests, mode, batch_size):
    """Use the same collate/backbone/heads as fast=False, also capturing the exact centre margins."""
    direction = model._check_mode(mode)
    if mode == "image_first":
        groups = [(0, len(requests))]
    else:
        groups = [(a, min(a + batch_size, len(requests))) for a in range(0, len(requests), batch_size)]
    for a, z in groups:
        batch = {**b, **{k: b[k][a:z] for k in ("q_img", "txt_ids", "txt_valid")},
                 "txt_pos": b["txt_pos"][:, a:z]}
        o = model.net(batch, feats, direction)
        rv, rc = b["vis_valid"], b["centers"]
        if direction == "t2i":
            rv, rc = rv[batch["q_img"]], rc[batch["q_img"]]
        out = model.heads(o["r"], o["vis"], rv, rc, o["vis_index"])
        multi = model.heads.decode_multi(o["r"], o["vis"], rv, rc, o["vis_index"], out)
        p = out["presence_logit"].sigmoid().tolist()
        piou = out["iou_logit"].sigmoid().tolist()
        probs, inds = out["center_logits"].topk(2)
        scale = torch.tensor(it["orig_wh"] * 2)
        boxes = out["box_pred"].clamp(0, 1).cpu() * scale
        for i, ((mb, ms), box, pp, pi, logits, indices) in enumerate(
            zip(multi, boxes.tolist(), p, piou, probs.tolist(), inds.tolist())
        ):
            result = Result(requests[a + i], pp, box, pi, (mb * scale).tolist(), ms.tolist())
            yield result, dict(center_indices=indices, center_logits=logits)


@torch.no_grad()
def predict(model, root, out_path, exact, batch_size, modes=MODES, resume=False, limit=None):
    by_file = gate_requests(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_suffix(".partial.jsonl")
    saved = {}
    if resume and partial.exists():
        # Retain only completed rows. An interruption may leave an incomplete last line.
        with partial.open() as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    break
                saved[tuple(row["key"]), row["mode"]] = row
    with partial.open("w") as f:
        for number, (name, rows) in enumerate(by_file.items(), 1):
            if limit and number > limit:
                break
            image = str(root / "images" / name)
            requests = [r["request"] for r in rows]
            if exact and any((tuple(row["key"]), mode) not in saved for row in rows for mode in modes):
                it = model._item(image, requests, None)
                b = to_dev(collate([it], model.tok, model.cfg, "i2t"), model.device)
                feats = model.net.image_features(b["pixel_values"], b["grid_thw"])
            for mode in modes:
                if all((tuple(row["key"]), mode) in saved for row in rows):
                    for row in rows:
                        print(json.dumps(saved[tuple(row["key"]), mode]), file=f)
                    continue
                if exact:
                    preds = reference(model, it, b, feats, requests, mode, batch_size)
                elif mode == "cached":
                    bank = model.encode([image])
                    preds = (model._mac.search_bank(request, bank, 0.5, diagnostics=True)[0] for request in requests)
                else:
                    n = batch_size if mode == "text_first" else len(rows)
                    preds = (pair for a in range(0, len(rows), n)
                             for pair in model._mac.ask(image, requests[a:a+n], model._check_mode(mode), 0.5, diagnostics=True))
                for row, (result, diagnostics) in zip(rows, preds, strict=True):
                    print(json.dumps(dict(row, mode=mode, **asdict(result), **diagnostics), allow_nan=False), file=f)
            f.flush()
            if number % 5 == 0:
                print(f"{number}/{len(by_file)} images", flush=True)
    partial.replace(out_path)


def metrics(rows, name):
    records = []
    for r in rows:
        W, H = r["wh"]
        scale = np.array([W, H, W, H])
        records.append(dict(present=r["present"], n_targets=len(r["gt"]), p_present=r["present_prob"],
                            box=(r["box"] / scale).tolist(), multi=[(b / scale).tolist() for b in r["boxes"]], gt=r["gt"]))
    scores = summarize(records)
    return scores["acc50" if name == "refcoco_val" else "multi_f1"] * 100


def compare(ref, candidate, modes, allow_margin, limited=False):
    lookup = {(tuple(r["key"]), r["mode"]): r for r in candidate}
    if len(lookup) != len(candidate):
        raise ValueError("Candidate keys must be unique")
    ok = True
    for mode in modes:
        ref_mode = "image_first" if mode == "cached" else mode
        old = [r for r in ref if r["mode"] == ref_mode]
        new_keys = {k for k, m in lookup if m == mode}
        if limited:
            old = [r for r in old if tuple(r["key"]) in new_keys]
        if not old or set(tuple(r["key"]) for r in old) != new_keys:
            raise ValueError(f"{mode}: reference and candidate requests differ")
        new = [lookup[tuple(r["key"]), mode] for r in old]
        presence = confidence = edges = violations = ties = counts = 0
        for a, b in zip(old, new):
            dp = abs(a["present_prob"] - b["present_prob"])
            dc = abs(a["present_prob"] * a["iou_prob"] - b["present_prob"] * b["iou_prob"])
            de = max(abs(x - y) for x, y in zip(a["box"], b["box"]))
            presence, confidence = max(dp, presence), max(dc, confidence)
            counts += len(a["boxes"]) == len(b["boxes"])
            if a["present_prob"] > 0.5:
                edges = max(de, edges)
                if de > 2:
                    margin = a["center_logits"][0] - a["center_logits"][1]
                    old_idx, new_idx = a["center_indices"][0], b["center_indices"][0]
                    allowed = allow_margin is not None and margin <= allow_margin and new_idx == a["center_indices"][1]
                    violations += not allowed
                    ties += allowed
                    status = "Accepted near-tie" if allowed else "Edge violation"
                    print(f"{status} {mode} {a['key']} Δ={de:.3f} px, exact margin={margin:.5f}, "
                          f"centre={old_idx}→{new_idx}, exact={a['box']}, candidate={b['box']}, request={a['request']!r}")
            if dp > 0.02 or dc > 0.02:
                print(f"Probability violation {mode} {a['key']} presence Δ={dp:.5f}, confidence Δ={dc:.5f}, "
                      f"centre={a['center_indices']}→{b['center_indices']}, request={a['request']!r}")
        ratio = counts / len(old)
        ok &= presence <= 0.02 and confidence <= 0.02 and violations == 0 and ratio >= 0.98
        print(f"| {mode} | {len(old)} | {presence:.5f} | {confidence:.5f} | {edges:.3f} | "
              f"{violations} | {ties} | {counts}/{len(old)} ({ratio:.2%}) |")
        for name in SETS:
            seqs = [[r for r in seq if r["key"][0] == name] for seq in (old, new)]
            a, b = (metrics(seq, name) for seq in seqs)
            ok &= abs(a - b) <= 0.5
            print(f"{mode} {name}: {a:.3f} → {b:.3f}, Δ {b-a:+.3f} points")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt")
    ap.add_argument("--base")
    ap.add_argument("--gate", type=Path)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--save-reference", action="store_true")
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--text-batch", type=int, default=16)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--mode", choices=CANDIDATE_MODES, action="append")
    ap.add_argument("--near-tie-margin", type=float, help="allow a listed top-two centre flip within this exact logit margin")
    args = ap.parse_args()
    modes = args.mode or (MODES if args.save_reference else CANDIDATE_MODES)
    if args.save_reference and "cached" in modes:
        ap.error("cached predictions are compared to the exact image_first path")
    if not args.save_reference and not args.output:
        ap.error("--output is required for a gate comparison")
    if args.text_batch < 1:
        ap.error("--text-batch must be positive")
    if not args.compare_only:
        if not args.ckpt or not args.gate:
            ap.error("--ckpt and --gate are required for inference")
        gate_requests(args.gate)
        model = GlanceX.from_pretrained(args.ckpt, base=args.base, device="mps", fast=not args.save_reference)
        if not args.save_reference and model._mac is None:
            ap.error("MLX is required for candidate inference")
        predict(model, args.gate, args.reference if args.save_reference else args.output,
                args.save_reference, args.text_batch, modes, args.resume, args.limit)
    if not args.save_reference:
        print("| Direction | Requests | Max Δ presence | Max Δ confidence | Max Δ edge (px) | Violations | Ties | Count match |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        ref, candidate = read_rows(args.reference), read_rows(args.output)
        if not compare(ref, candidate, modes, args.near_tie_margin, limited=bool(args.limit)):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
