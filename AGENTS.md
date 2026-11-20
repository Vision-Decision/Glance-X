# AGENTS.md

Notes for coding agents and contributors working in this repo.

## What this is

This repository is the inference and evaluation package for Glance-X-Lite, a 0.8B referring-expression box model
(Qwen3.5-0.8B-Base) with two directions, `image_first` (with an image cache) and `text_first`. Start with
`docs/ARCHITECTURE.md`. Training and data code live in Glanece-Training, not here.

## Layout

- `glancex/inference.py`: the public API (`GlanceX`, `Result`). Keep it small and stable.
- `glancex/backbone.py` and `glancex/_layers.py`: the prefix/block computation. Any change here must keep
  `scripts/test_equivalence.py` printing `ALL_OK`. That test needs a CUDA GPU.
- `glancex/heads.py` and `glancex/_box.py`: heads and losses. Geometry is normalised to [0, 1].
- `glancex/data.py`: records → batches. Streams are LEFT padded. Text positions start at `cfg.text_pos0`.
- `glancex/config.py`: one flat dataclass. New keys need defaults, because old checkpoints must keep loading
  (`update` ignores unknown keys).

## Checks

```bash
pip install -e ".[dev]"
ruff check glancex scripts demo tests
pytest -q                                      # CPU only, no weights needed
python scripts/test_equivalence.py --model_id Qwen/Qwen3.5-0.8B-Base   # GPU
```

## Rules

- Never commit weights (`*.pt`, `*.safetensors`), data, tokens, or machine-specific paths. CI greps for keys.
- Every number in the README must come from a real run. The demo video is rendered from real predictions
  (`demo/predictions.json`).
- Weights are published on Hugging Face (`Vision-Decision/Glance-X-Lite`); the model card is `hf/README.md`.
  Exporting a checkpoint is done in Glanece-Training (`scripts/export_hf.py`).
