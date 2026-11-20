---
license: cc-by-nc-sa-4.0
base_model: Qwen/Qwen3.5-0.8B-Base
base_model_relation: adapter
pipeline_tag: zero-shot-object-detection
language:
  - en
  - zh
datasets:
  - Vision-Decision/Glance-COCO
tags:
  - visual-grounding
  - referring-expression-comprehension
  - open-vocabulary-detection
  - multi-target-grounding
  - qwen3.5
  - lora
  - safetensors
---

<p align="center">
  <img src="assets/banner.png" width="720" alt="Glance-X-Lite">
</p>

<p align="center">
  <a href="https://github.com/Vision-Decision/Glance-X"><img src="https://img.shields.io/badge/GitHub-Glance--X-181717?logo=github&logoColor=white" alt="GitHub: Glance-X"></a>
  <a href="https://github.com/Vision-Decision/Glanece-Training"><img src="https://img.shields.io/badge/training-Glanece--Training-181717?logo=github&logoColor=white" alt="Training code"></a>
  <a href="https://github.com/Vision-Decision/Glance-Data"><img src="https://img.shields.io/badge/data-Glance--Data%20(GitHub)-181717?logo=github&logoColor=white" alt="Glance-Data on GitHub"></a>
  <a href="https://huggingface.co/datasets/Vision-Decision/Glance-COCO"><img src="https://img.shields.io/badge/data-Glance--COCO%20(HF)-FFD21E?logo=huggingface&logoColor=000" alt="Glance-Data on Hugging Face"></a>
  <a href="https://huggingface.co/Qwen/Qwen3.5-0.8B-Base"><img src="https://img.shields.io/badge/base-Qwen3.5--0.8B--Base-7C3AED" alt="Base model"></a>
  <img src="https://img.shields.io/badge/weights-CC%20BY--NC--SA%204.0-6B7280" alt="Weights license">
</p>

**Glance-X-Lite** finds objects in an image from a text request, in English or Chinese. It answers with a box
around every match, or says the object is not in the image. It is a 0.8B model with two directions:

| Direction | Cached | Best for |
|---|---|---|
| `image_first` | the image: encode once, then each request costs ~30 tokens | one image, many requests; a photo library searched again and again |
| `text_first` | the request; each new image reads it | one request over new images (camera, video); best multi-target recall |

<p align="center">
  <img src="assets/demo.gif" width="820" alt="Demo: one request over 256 cached images picks an image, then new requests find each target in it">
  <br><sub>Real outputs: one request over 256 cached images (68 ms on one GPU), then new requests inside the picked image.</sub>
</p>

## Use it

The code is on GitHub: **[Vision-Decision/Glance-X](https://github.com/Vision-Decision/Glance-X)**.

```bash
pip install "glancex[fla] @ git+https://github.com/Vision-Decision/Glance-X"     # drop [fla] without a CUDA GPU
```

```python
from glancex import GlanceX

m = GlanceX.from_pretrained("Vision-Decision/Glance-X-Lite")   # this repo + Qwen/Qwen3.5-0.8B-Base

m.ask("tennis.jpg", ["the kid with glasses in the middle", "左边举起球拍的男孩", "a frisbee"])
# Result('the kid with glasses in the middle', present_prob=1.000, box=[250, 153, 369, 422], confidence=0.991, n_boxes=1)
# Result('左边举起球拍的男孩', present_prob=0.983, box=[62, 146, 260, 427], confidence=0.979, n_boxes=1)
# Result('a frisbee', present_prob=0.000, box=[511, 134, 588, 261], confidence=0.000, n_boxes=0)

bank = m.encode(photos)                                        # image cache, ~25 MB per image
m.search("a giraffe eating from someone's hand", bank)[0]
m.search("a giraffe eating from someone's hand", photos, mode="text_first")[0]
```

## Files

| File | Contents |
|---|---|
| `adapter_model.safetensors` | LoRA r=64 on every language-model linear layer of Qwen3.5-0.8B-Base (fp32, 164 MB) |
| `heads.safetensors` | presence, box (patch centre + 4×64-bin edges), IoU and multi-box objectness heads (25 MB) |
| `glancex_config.json` | model config (24 layers, K=2, positions, head sizes) |

## How it works

<p align="center"><img src="assets/architecture.png" width="860" alt="Glance-X-Lite architecture"></p>

- **Inputs inside the language model.** The image tokens go inside the language model and are fused with the
  request over 22 layers.
- **`image_first`:** the image is a shared prefix that never reads a request, so it can be cached.
- **`text_first`:** the request is the prefix, and the image tokens read it.
- **Heads:** presence, one box (patch centre + four edge distributions), IoU, and per-patch objectness for every
  target.

## Results

Full splits, Acc@0.5 (%):

| | `image_first` | `text_first` |
|---|:-:|:-:|
| RefCOCO val / testA / testB | 90.8 / 92.7 / 87.8 | 89.9 / 91.8 / 86.1 |
| RefCOCO+ val / testA / testB | 85.3 / 90.2 / 80.2 | 83.1 / 88.0 / 76.5 |
| RefCOCOg val / test | 87.5 / 87.7 | 85.9 / 85.5 |
| **8-split mean** | **87.8** | **85.9** |
| gRefCOCO val: AUROC of P(present) / multi-target F1 | 97.4 / 75.0 | 96.4 / **76.1** |
| Glance-Data: Acc@0.5 / multi-target F1 | 86.4 / 59.6 | 85.8 / **64.6** |

Latency on one H200, end to end (ms, N = 1 / 16 / 64 / 128):

| Workload | Latency |
|---|---|
| one image, N requests | 36 / 44 / 64 / 94 |
| one request, N cached images | 34 / 33 / 36 / 42 |
| one request, N new images (`text_first`) | 38 / 181 / 658 / 1,293 |

For comparison, the 4B Glance-X2 scores 91.3 on the RefCOCO-family mean. Full tables are in
[RESULTS.md](https://github.com/Vision-Decision/Glance-X/blob/main/docs/RESULTS.md).

## Training

- **Starting point:** Qwen3.5-0.8B-Base. The vision tower is frozen, and LoRA r=64 sits on the language model.
- **Schedule:** 2 epochs (6,100 steps at 8 images per step), half of the batches in each direction, about 75 minutes
  on one H200.
- **Data:** the RefCOCO family plus gRefCOCO negatives (355k requests) and
  [Glance-Data](https://github.com/Vision-Decision/Glance-Data) (1.83M requests).
- **Code:** [Glanece-Training](https://github.com/Vision-Decision/Glanece-Training), `configs/lite/glance-x-lite.json`.

## Limitations and license

- The model was trained on natural photos (COCO) only. `joint` is not supported.
- The weights are licensed **CC BY-NC-SA 4.0**, for non-commercial use only, because the training data is
  non-commercial. The code is Apache-2.0.
