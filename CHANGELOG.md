# Changelog

## 0.1.0 (2026-11-19)

- First release of Glance-X-4B (K=8):
  - one Qwen3.5-4B-Base backbone with three directions: `image_first`, `text_first` and `joint`
  - a multi-box objectness head
- `GlanceX` API: `from_pretrained` / `ask` / `find` / `search`, loading from Hugging Face, an exported folder, or a
  training checkpoint.
- Training and per-direction evaluation, with results reported separately for the original benchmarks and
  Glance-Data.
- Weights published at [Vision-Decision/Glance-X-4B](https://huggingface.co/Vision-Decision/Glance-X-4B).
