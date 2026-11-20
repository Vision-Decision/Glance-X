"""One flat config. Defaults = Glance-X-4B (K=8). Override with a JSON file and/or `--set key=value`."""
import argparse
import dataclasses
import json
from dataclasses import dataclass


@dataclass
class Config:
    # --- model ---
    model_id: str = "Qwen/Qwen3.5-4B-Base"
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    # LoRA on every LLM linear layer; the vision tower stays frozen
    lora_target: str = (r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|"
                        r"gate_proj|up_proj|down_proj)")
    grad_ckpt: bool = True
    bidir_k: int = 8                     # modality-stage layers before fusion (0 = fuse from the first layer)
    bidir_block_mask: str = "block_bidir"  # attention inside a fusion block: block_bidir | causal (tests)
    text_pos0: int = 66                  # fixed first position of the text stream (> 1 + largest image grid side)
    # --- heads ---
    head_dim: int = 512
    edge_bins: int = 64
    edge_lo: float = -0.1                # edge offsets (fraction of the image side) from the chosen patch centre
    edge_hi: float = 1.0
    edge_sigma_bins: float = 1.0
    w_presence: float = 1.0
    w_center: float = 1.0
    w_cramer: float = 0.1
    w_giou: float = 2.0
    w_iou_head: float = 0.5
    w_obj: float = 1.0                   # multi-box objectness
    obj_shrink: float = 0.5              # objectness positive: patch centre inside the central share of a target box
    max_targets: int = 16
    # --- data ---
    train_file: str = "data/train.jsonl"
    image_root: str = "data/images"
    max_vis_tokens: int = 640            # visual tokens per image (one per 32x32 px)
    max_upscale: float = 2.0
    max_q_per_record: int = 48
    max_q_tokens: int = 40
    min_orig_frac: float = 0.34          # at least this share of a record's sampled requests come from original data
    limit_records: int = 0
    # --- directions during training ---
    mode_probs: str = "0.5,0.25,0.25"    # image_first, text_first, joint share of batches
    i2t_q_per_image: int = 24            # requests per image in image_first batches
    pair_q_per_image: int = 3            # (request, image) pairs per image in text_first / joint batches
    # fixed sequence lengths (fla's Hopper kernels compile per shape); 0 = batch maximum
    pad_img_to: int = 642
    pad_txt_to: int = 44
    pad_txt_block_to: int = 44
    pad_t2i_block_to: int = 644
    pad_bi_block_to: int = 688
    # --- optimisation ---
    images_per_step: int = 8
    grad_accum: int = 1
    epochs: float = 2.0
    max_steps: int = 0
    lr: float = 2e-4
    lr_heads: float = 5e-4
    weight_decay: float = 0.0
    warmup_steps: int = 100
    min_lr_ratio: float = 0.05
    clip: float = 1.0
    num_workers: int = 8
    seed: int = 0
    log_every: int = 20
    save_every: int = 500
    out_dir: str = "outputs/glance-x-4b-k8"
    resume: str = ""
    init_ckpt: str = ""

    @staticmethod
    def parse(argv=None):
        ap = argparse.ArgumentParser()
        ap.add_argument("--cfg", default="")
        ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
        a = ap.parse_args(argv)
        c = Config()
        if a.cfg:
            c = c.update(json.load(open(a.cfg)))
        return c.update(dict(s.split("=", 1) for s in a.set), from_str=True), a

    def update(self, d, from_str=False, strict=False):
        """Return a copy with `d` applied. Unknown keys raise if strict, else are ignored (old checkpoints)."""
        fields = {f.name for f in dataclasses.fields(self)}
        new = dataclasses.asdict(self)
        for k, v in d.items():
            if k not in fields:
                if strict:
                    raise KeyError(f"unknown config key {k}")
                continue
            if from_str:
                t = type(new[k])
                v = (v.lower() in ("1", "true", "yes")) if t is bool else t(v)
            new[k] = v
        return Config(**new)

    def to_json(self):
        return json.dumps(dataclasses.asdict(self), indent=1)
