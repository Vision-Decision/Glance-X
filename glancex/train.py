"""Train Glance-X (docs/ARCHITECTURE.md).

    python -m glancex.train --cfg configs/glance-x-4b-k8.json [--set key=value ...]
"""
import json, math, os, random, time
import torch

from ._box import center_target_box
from .config import Config
from .data import RecordDataset, collate_train
from .heads import loss_fn, obj_target, union_center_target
from .build import build, load_ckpt, save_ckpt


def to_dev(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def forward_batch(net, heads, batch, train=True):
    """Forward in batch['mode']; returns head outputs, loss targets and the per-question geometry."""
    feats = net.image_features(batch["pixel_values"], batch["grid_thw"])
    dev = batch["img_ids"].device.type
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        o = net(batch, feats, batch["mode"])
    q_img = batch["q_img"]
    if batch["mode"] == "i2t":   # vis rows = images
        rows_valid, rows_centers = batch["vis_valid"], batch["centers"]
    else:                        # vis rows = one per question (its own text-conditioned image)
        rows_valid, rows_centers = batch["vis_valid"][q_img], batch["centers"][q_img]
    valid_q, centers_q, grid_q = batch["vis_valid"][q_img], batch["centers"][q_img], batch["grid_hw"][q_img]
    tgt = sel = None
    if train:
        present = batch["present"]
        ct_sel = center_target_box(batch["box_sel"], centers_q, valid_q, grid_q)
        sel = torch.multinomial(ct_sel + (~present)[:, None] * valid_q.float() + 1e-12, 1)[:, 0]
        tgt = dict(present=present, boxes=batch["boxes"], n_boxes=batch["n_boxes"], box_sel=batch["box_sel"],
                   center_t=union_center_target(batch["boxes"], batch["n_boxes"], centers_q, valid_q, grid_q),
                   obj_t=obj_target(batch["boxes"], batch["n_boxes"], centers_q, valid_q, net.cfg.obj_shrink),
                   valid=valid_q)
    out = heads(o["r"], o["vis"], rows_valid, rows_centers, o["vis_index"], sel_idx=sel, q_tok=o["q_tok"],
                q_valid=o["q_valid"], r_idx=o["r_idx"])
    return out, tgt, (o, rows_valid, rows_centers)


def main():
    cfg, _ = Config.parse()
    torch.manual_seed(cfg.seed); random.seed(cfg.seed)
    torch.backends.cuda.enable_cudnn_sdp(False)
    os.makedirs(cfg.out_dir, exist_ok=True)
    open(f"{cfg.out_dir}/config.json", "w").write(cfg.to_json())
    dev = "cuda"
    t0 = time.time()
    net, heads, proc = build(cfg, dev)
    net.train(); heads.train()
    print(f"[load] {time.time() - t0:.1f}s K={cfg.bidir_k} layers={len(net.layer_types)} hidden={net.hidden}", flush=True)
    ds = RecordDataset(cfg.train_file, cfg.image_root, proc, cfg, limit=cfg.limit_records)
    tok = ds.tok
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.images_per_step, shuffle=True, drop_last=True,
                                     num_workers=cfg.num_workers, collate_fn=lambda it: collate_train(it, tok, cfg),
                                     persistent_workers=cfg.num_workers > 0, pin_memory=True)
    steps_per_epoch = math.ceil(len(dl) / cfg.grad_accum)
    total = cfg.max_steps or int(steps_per_epoch * cfg.epochs)
    lora_p = [p for n, p in net.hf.named_parameters() if p.requires_grad]
    print(f"[params] lora={sum(p.numel() for p in lora_p) / 1e6:.1f}M heads={sum(p.numel() for p in heads.parameters()) / 1e6:.1f}M "
          f"records={len(ds)} steps={total}", flush=True)
    groups = [dict(params=lora_p, lr=cfg.lr), dict(params=list(heads.parameters()), lr=cfg.lr_heads)]
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.95), fused=True)
    base_lrs = [g["lr"] for g in opt.param_groups]

    def lr_factor(s):
        if s < cfg.warmup_steps:
            return (s + 1) / cfg.warmup_steps
        p = (s - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    step = 0
    if cfg.init_ckpt and not cfg.resume:
        load_ckpt(cfg.init_ckpt, net, heads)
        print(f"[init] {cfg.init_ckpt}", flush=True)
    if cfg.resume:
        sd = load_ckpt(cfg.resume, net, heads)
        step = sd["extra"].get("step", 0)
        if "opt" in sd["extra"]:
            opt.load_state_dict(sd["extra"]["opt"])
        print(f"[resume] {cfg.resume} step {step}", flush=True)
    logf = open(f"{cfg.out_dir}/train_log.jsonl", "a")
    torch.cuda.reset_peak_memory_stats()
    micro, acc, n_acc, tok_acc, t_step = 0, {}, {}, 0, time.time()
    t_train = time.time()
    done = False
    while not done:
        for batch in dl:
            batch = to_dev(batch, dev)
            mode = batch["mode"]
            out, tgt, _ = forward_batch(net, heads, batch, train=True)
            loss, parts = loss_fn(out, tgt, cfg)
            (loss / cfg.grad_accum).backward()
            micro += 1
            tok_acc += batch["n_tokens"]
            for k, v in list(parts.items()) + [("loss", loss)]:
                key = f"{mode}/{k}"
                acc[key] = acc.get(key, 0.0) + float(v.detach() if torch.is_tensor(v) else v)
                n_acc[key] = n_acc.get(key, 0) + 1
            if micro % cfg.grad_accum:
                continue
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * lr_factor(step)
            gn = torch.nn.utils.clip_grad_norm_(lora_p + list(heads.parameters()), cfg.clip)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.log_every == 0 or step == 1:
                torch.cuda.synchronize()
                dt = time.time() - t_step
                n_steps = cfg.log_every if step > 1 else 1
                rec = {k: v / n_acc[k] for k, v in acc.items()}
                rec.update(step=step, total=total, lr=opt.param_groups[0]["lr"], gnorm=float(gn),
                           s_per_step=dt / n_steps, tok_per_s=tok_acc / dt, mem_gb=torch.cuda.max_memory_allocated() / 2**30,
                           elapsed_min=(time.time() - t_train) / 60)
                print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in sorted(rec.items())}), flush=True)
                logf.write(json.dumps(rec) + "\n"); logf.flush()
                acc, n_acc, tok_acc, t_step = {}, {}, 0, time.time()
            if cfg.save_every and step % cfg.save_every == 0:
                save_ckpt(f"{cfg.out_dir}/ckpt_last.pt", net, heads, cfg, dict(step=step))
            if step >= total:
                done = True
                break
    save_ckpt(f"{cfg.out_dir}/ckpt_final.pt", net, heads, cfg, dict(step=step))
    print(f"[done] {step} steps in {(time.time() - t_train) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
