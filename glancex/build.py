"""Load the Qwen3.5 base, inject LoRA into the language model, attach the Glance-X heads; checkpoint I/O."""
import os

import torch
from peft import LoraConfig, get_peft_model_state_dict, inject_adapter_in_model, set_peft_model_state_dict
from transformers import AutoModelForImageTextToText, AutoProcessor

from .backbone import GlanceXBackbone
from .heads import GlanceXHeads


def load_base(cfg, device="cuda", dtype=torch.bfloat16):
    model = AutoModelForImageTextToText.from_pretrained(cfg.model_id, dtype=dtype, attn_implementation="sdpa").to(device)
    proc = AutoProcessor.from_pretrained(cfg.model_id)
    return model, proc


def build(cfg, device="cuda", lora=True, dtype=torch.bfloat16):
    """-> (backbone, heads, processor). LoRA weights are fp32 masters; the forward runs under bf16 autocast on CUDA."""
    hf, proc = load_base(cfg, device, dtype)
    hf.requires_grad_(False)
    hf.config.use_cache = False
    if lora and cfg.lora_r > 0:
        lc = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                        target_modules=cfg.lora_target, bias="none")
        inject_adapter_in_model(lc, hf)
        for n, p in hf.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)
                p.data = p.data.float()
        for m in hf.modules():          # let autocast handle the matmul instead of up-casting every activation
            if hasattr(m, "lora_A"):
                m.cast_input_dtype_enabled = False
    net = GlanceXBackbone(hf, cfg)
    heads = GlanceXHeads(net.hidden, cfg).to(device)   # fp32
    return net, heads, proc


def trainable_state(net, heads):
    return dict(lora=get_peft_model_state_dict(net.hf), heads=heads.state_dict())


def save_ckpt(path, net, heads, cfg, extra=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(dict(**trainable_state(net, heads), cfg=cfg.__dict__, extra=extra or {}), path)


def load_ckpt(path_or_state, net, heads):
    """path_or_state: a training checkpoint (.pt) or a dict with "lora" and "heads" (see load_state)."""
    sd = path_or_state if isinstance(path_or_state, dict) else torch.load(path_or_state, map_location="cpu",
                                                                           weights_only=False)
    if sd.get("lora"):
        res = set_peft_model_state_dict(net.hf, sd["lora"])
        bad = [k for k in getattr(res, "unexpected_keys", []) if "lora" in k]
        assert not bad, f"unexpected LoRA keys: {bad[:5]}"
    heads.load_state_dict(sd["heads"])
    return sd


CONFIG_FILE, LORA_FILE, HEADS_FILE = "glancex_config.json", "adapter_model.safetensors", "heads.safetensors"


def load_state(src, revision=None):
    """Weights + config from a training checkpoint (.pt), an exported folder, or a Hugging Face repo id.
    Returns dict(cfg=..., lora=..., heads=...)."""
    import json

    from safetensors.torch import load_file
    if os.path.isfile(src):
        return torch.load(src, map_location="cpu", weights_only=False)
    if not os.path.isdir(src):
        from huggingface_hub import snapshot_download
        src = snapshot_download(src, revision=revision, allow_patterns=[CONFIG_FILE, LORA_FILE, HEADS_FILE])
    return dict(cfg=json.load(open(os.path.join(src, CONFIG_FILE))), lora=load_file(os.path.join(src, LORA_FILE)),
                heads=load_file(os.path.join(src, HEADS_FILE)))


def export_state(sd, out_dir, cfg_overrides=None):
    """Write an exported folder (config JSON + two safetensors files) from a training checkpoint dict."""
    import json

    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    cfg = dict(sd["cfg"], **(cfg_overrides or {}))
    json.dump(cfg, open(os.path.join(out_dir, CONFIG_FILE), "w"), indent=1)
    save_file({k: v.contiguous() for k, v in sd["lora"].items()}, os.path.join(out_dir, LORA_FILE))
    save_file({k: v.contiguous() for k, v in sd["heads"].items()}, os.path.join(out_dir, HEADS_FILE))
