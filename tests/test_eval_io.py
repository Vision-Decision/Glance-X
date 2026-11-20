"""Metrics and the exported-weights round trip (CPU)."""
import torch

import json

from safetensors.torch import save_file

from glancex.build import load_state
from glancex.evaluate import auroc, f1_sets


def test_auroc():
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert abs(auroc([0.5, 0.5], [1, 0]) - 0.5) < 1e-9


def test_f1_sets():
    gt = [[0, 0, 10, 10], [20, 20, 30, 30]]
    assert f1_sets(gt, gt) == 1.0
    assert f1_sets([], gt) == 0.0
    assert abs(f1_sets([[0, 0, 10, 10]], gt) - 2 / 3) < 1e-9
    assert abs(f1_sets([[0, 0, 10, 10], [0, 0, 10, 10]], gt) - 0.5) < 1e-9     # a duplicate is not a second hit


def test_load_exported_folder(tmp_path):
    lora, heads = {"a.lora_A.weight": torch.randn(4, 3)}, {"presence.0.weight": torch.randn(2, 2)}
    json.dump(dict(bidir_k=2, model_id="Qwen/Qwen3.5-0.8B-Base"), open(tmp_path / "glancex_config.json", "w"))
    save_file(lora, str(tmp_path / "adapter_model.safetensors"))
    save_file(heads, str(tmp_path / "heads.safetensors"))
    back = load_state(str(tmp_path))
    assert back["cfg"]["bidir_k"] == 2
    assert torch.equal(back["lora"]["a.lora_A.weight"], lora["a.lora_A.weight"])
    assert torch.equal(back["heads"]["presence.0.weight"], heads["presence.0.weight"])
