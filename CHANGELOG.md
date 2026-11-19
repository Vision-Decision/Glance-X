# Changelog

## 0.2.0 (2026-11-19): Glance-X2

- **Glance-X2.** One more epoch of `image_first` training from Glance-X K=8 (`configs/glance-x2-4b.json`). It
  scores 91.3 on the RefCOCO-family mean (Glance-X: 90.7 `image_first`, 90.9 `joint`) and 80.6 multi-target F1 on
  gRefCOCO val (Glance-X: 77.3 `image_first`, 82.7 `joint`). The weights are not released yet.
- **Image cache.** `GlanceX.encode(images)` returns an `ImageBank`, and `search(request, bank)` runs only the
  request's block per image. Searching 128 cached images takes 71 ms on one H200. The cached path is exact against
  the full forward pass.
- **Faster inference by default** (`fast=True`):
  - LoRA merged into the base weights
  - images loaded on worker threads and normalised on the GPU
  - efficient masked attention
  - vectorised multi-box NMS
  - CUDA-graph replay of the language model and heads

  One image with one request: 148 → 58 ms. Use `fast=False` for the exact training-time computation.
- `devices=[...]`: one replica per GPU, with batches split across them.
- Optional two-way decoder (`decoder_layers`). It brought no gain in our runs, so it is off by default.
- Scripts: profiling, latency benchmarks, a fast-vs-exact check, and the comparison with the TypeSafe Jev API.

## 0.1.0 (2026-11-19)

- First release of Glance-X-4B (K=8):
  - one Qwen3.5-4B-Base backbone with three directions: `image_first`, `text_first` and `joint`
  - a multi-box objectness head
- `GlanceX` API: `from_pretrained` / `ask` / `find` / `search`, loading from Hugging Face, an exported folder, or a
  training checkpoint.
- Training and per-direction evaluation, with results reported separately for the original benchmarks and
  Glance-Data.
- Weights published at [Vision-Decision/Glance-X-4B](https://huggingface.co/Vision-Decision/Glance-X-4B).
