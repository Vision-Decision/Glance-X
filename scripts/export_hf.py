"""Export a training checkpoint to the Hugging Face layout loaded by GlanceX.from_pretrained.

python scripts/export_hf.py --ckpt outputs/glance-x-4b-k8/ckpt_final.pt --out hf/Glance-X-4B

Writes glancex_config.json, adapter_model.safetensors (LoRA) and heads.safetensors. Machine-specific paths in the
training config are reset to the repo defaults, and model_id is set to the public base model.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glancex.build import export_state  # noqa: E402
from glancex.config import Config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--base", default="Qwen/Qwen3.5-4B-Base")
a = ap.parse_args()
sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
known = Config().__dict__
cfg = {k: v for k, v in sd["cfg"].items() if k in known}            # drop keys of older code versions
d = Config()
over = dict(model_id=a.base, train_file=d.train_file, image_root=d.image_root, out_dir=d.out_dir, resume="",
            init_ckpt="")
export_state(dict(sd, cfg=cfg), a.out, over)
print(f"step {sd.get('extra', {}).get('step')} -> {a.out}: {sorted(os.listdir(a.out))}")
