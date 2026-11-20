# Changelog

## 0.3.0 (2026-11-20): Glance-X-Lite

- **Glance-X-Lite** (0.8B, Qwen3.5-0.8B-Base) is now the released model of this repository, at
  [Vision-Decision/Glance-X-Lite](https://huggingface.co/Vision-Decision/Glance-X-Lite).
  - Two directions: `image_first`, with the image cache, and `text_first`.
  - RefCOCO-family mean 87.8 (`image_first`); gRefCOCO val multi-target F1 76.1 (`text_first`).
- This repository is now inference and evaluation only. Training, data preparation, the 4B models (Glance-X, X2,
  Flash) and all experiments moved to
  [Glanece-Training](https://github.com/Vision-Decision/Glanece-Training).
- `evaluate` accepts a Hugging Face id or an exported folder (`--ckpt`) and an image folder (`--image_root`).
- The API refuses directions a checkpoint was not trained in. For Lite, that is `joint`.

## 0.2.0 (2026-11-19): Glance-X2

- Image cache (`encode` / `search(request, bank)`), the fast inference path, multi-GPU, and Glance-X2 results.

## 0.1.0 (2026-11-19)

- First release of Glance-X-4B (three directions).
