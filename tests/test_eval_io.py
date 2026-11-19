"""Metrics and the exported-weights round trip (CPU)."""
import torch

from glancex.build import export_state, load_state
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


def test_export_roundtrip(tmp_path):
    sd = dict(cfg=dict(bidir_k=8, model_id="Qwen/Qwen3.5-4B-Base"),
              lora={"a.lora_A.weight": torch.randn(4, 3)}, heads={"presence.0.weight": torch.randn(2, 2)})
    export_state(sd, tmp_path, dict(out_dir="outputs/x"))
    back = load_state(str(tmp_path))
    assert back["cfg"]["bidir_k"] == 8 and back["cfg"]["out_dir"] == "outputs/x"
    assert torch.equal(back["lora"]["a.lora_A.weight"], sd["lora"]["a.lora_A.weight"])
    assert torch.equal(back["heads"]["presence.0.weight"], sd["heads"]["presence.0.weight"])
