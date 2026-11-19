# AGENTS.md

Notes for coding agents and contributors working in this repo.

## What this is

Glance-X is a referring-expression box model on Qwen3.5-4B-Base. It has one backbone with three directions
(`image_first` / `text_first` / `joint`). Start with `docs/ARCHITECTURE.md`.

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
python scripts/test_equivalence.py --model_id Qwen/Qwen3.5-4B-Base   # GPU
```

## Rules

- Never commit weights (`*.pt`, `*.safetensors`), data, tokens, or machine-specific paths. CI greps for keys.
- Every number in the README must come from a real run. The demo video is rendered from real predictions
  (`demo/predictions.json`).
- Weights are published on Hugging Face (`Vision-Decision/Glance-X-4B`) with `scripts/export_hf.py`. The model
  card is `hf/README.md`.
