"""Forward pass shared by the API and the evaluation: backbone in one direction, then the heads."""
import torch


def to_dev(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def forward_batch(net, heads, batch, train=False):
    """Forward in batch['mode'] -> head outputs, None (no training targets here), per-request geometry."""
    assert not train, "training lives in Glanece-Training (github.com/Vision-Decision/Glanece-Training)"
    feats = net.image_features(batch["pixel_values"], batch["grid_thw"])
    dev = batch["img_ids"].device.type
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        o = net(batch, feats, batch["mode"])
    q_img = batch["q_img"]
    if batch["mode"] == "i2t":   # vis rows = images
        rows_valid, rows_centers = batch["vis_valid"], batch["centers"]
    else:                        # vis rows = one per request (its own request-conditioned image)
        rows_valid, rows_centers = batch["vis_valid"][q_img], batch["centers"][q_img]
    out = heads(o["r"], o["vis"], rows_valid, rows_centers, o["vis_index"])
    return out, None, (o, rows_valid, rows_centers)
