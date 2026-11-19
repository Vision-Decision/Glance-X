---
license: cc-by-nc-sa-4.0
base_model: Qwen/Qwen3.5-4B-Base
base_model_relation: adapter
pipeline_tag: zero-shot-object-detection
language:
  - en
  - zh
datasets:
  - Vision-Decision/Glance-Data
tags:
  - visual-grounding
  - referring-expression-comprehension
  - open-vocabulary-detection
  - multi-target-grounding
  - bidirectional
  - qwen3.5
  - lora
  - safetensors
---

<p align="center">
  <img src="assets/banner.png" width="720" alt="Glance-X: one model, three ways to look">
</p>

<p align="center">
  <a href="https://github.com/Vision-Decision/Glance-X"><img src="https://img.shields.io/badge/GitHub-Glance--X-181717?logo=github&logoColor=white" alt="GitHub: Glance-X"></a>
  <a href="https://huggingface.co/Vision-Decision/Glance-X-4B"><img src="https://img.shields.io/badge/Hugging%20Face-Glance--X--4B-FFD21E?logo=huggingface&logoColor=000" alt="Hugging Face: Glance-X-4B"></a>
  <a href="https://github.com/Vision-Decision/Glance-Data"><img src="https://img.shields.io/badge/data-Glance--Data%20(GitHub)-181717?logo=github&logoColor=white" alt="Data on GitHub: Glance-Data"></a>
  <a href="https://huggingface.co/datasets/Vision-Decision/Glance-Data"><img src="https://img.shields.io/badge/data-Glance--Data%20(HF)-FFD21E?logo=huggingface&logoColor=000" alt="Data on Hugging Face: Glance-Data"></a>
  <a href="https://huggingface.co/Qwen/Qwen3.5-4B-Base"><img src="https://img.shields.io/badge/base-Qwen3.5--4B--Base-7C3AED" alt="Base model: Qwen3.5-4B-Base"></a>
  <img src="https://img.shields.io/badge/weights-CC%20BY--NC--SA%204.0-6B7280" alt="Weights license: CC BY-NC-SA 4.0">
</p>

**Glance-X-4B** finds objects in an image from a text request, such as *"the kid with glasses in the middle"* or
*"左边举起球拍的男孩"*. It draws a **box** around every matching object, or says **the object is not in the
image**.

**One set of weights runs in three directions**, and you pick one for each call:

| Direction | Computed once and shared | Best for |
|---|---|---|
| `image_first` &nbsp;image → text | the image | one image, many requests, answered in parallel |
| `text_first` &nbsp;text → image | the request | one request over many images or video frames |
| `joint` &nbsp;image ⇄ text | nothing: image and request read each other | the most accurate answer for one pair |

<p align="center">
  <img src="assets/demo.gif" width="820" alt="Demo: six requests on one image, multi-target answers, look-alike rejection, and one request searched over six images">
  <br><sub>Real outputs of this checkpoint on held-out images (MS COCO).</sub>
</p>

## Use it

The code lives on GitHub: **[Vision-Decision/Glance-X](https://github.com/Vision-Decision/Glance-X)**.

```bash
pip install "glancex[fla] @ git+https://github.com/Vision-Decision/Glance-X"     # drop [fla] without a CUDA GPU
```

```python
from glancex import GlanceX

m = GlanceX.from_pretrained("Vision-Decision/Glance-X-4B")   # this repo + Qwen/Qwen3.5-4B-Base

m.ask("tennis.jpg", ["the kid with glasses in the middle", "左边举起球拍的男孩", "a frisbee"])  # image_first
# Result('the kid with glasses in the middle', present_prob=1.000, box=[249, 165, 381, 424], confidence=0.997, n_boxes=1)
# Result('左边举起球拍的男孩', present_prob=0.992, box=[65, 143, 253, 422], confidence=0.987, n_boxes=1)
# Result('a frisbee', present_prob=0.000, box=[387, 93, 587, 330], confidence=0.000, n_boxes=0)

m.find("giraffes.jpg", "the three giraffes in the field", mode="joint").boxes
# [[47, 62, 242, 316], [333, 69, 515, 312], [522, 101, 638, 314]]

m.search("a giraffe eating from someone's hand", photos)[0]                                   # text_first, ranked
# Result("a giraffe eating from someone's hand", present_prob=0.994, box=[307, 22, 585, 324], confidence=0.753, n_boxes=0)
```

Every result has these fields:

- `present_prob`: P(the request matches something in the image)
- `box`: in pixels
- `iou_prob`: P(IoU ≥ 0.5)
- `confidence`: `present_prob × iou_prob`
- `boxes`: every target found by the multi-box head

## Files

| File | Contents |
|---|---|
| `adapter_model.safetensors` | LoRA r=64 on every language-model linear layer of Qwen3.5-4B-Base (fp32, 488 MB) |
| `heads.safetensors` | presence, box (patch centre + 4×64-bin edges), IoU and multi-box objectness heads (59 MB) |
| `glancex_config.json` | model config (K=8, positions, head sizes) |

The base model [Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) is downloaded separately
(about 9 GB). Its vision tower is used frozen.

## How it works

<p align="center"><img src="assets/architecture.png" width="860" alt="Glance-X architecture"></p>

1. **Modality stage** (layers 1–8). The image and the request run as separate streams, so each can be cached.
2. **Fusion stage** (layers 9–32). One stream becomes a shared prefix and the other a block that reads it. In
   `joint`, both sit in one block with attention running both ways.
3. **Heads.** Presence, single box, IoU, and per-patch objectness for multi-target requests.

With K=0 and a causal block mask, every direction equals a plain causal forward of the equivalent single sequence;
the [exactness test](https://github.com/Vision-Decision/Glance-X/blob/main/scripts/test_equivalence.py) checks this.
More in [ARCHITECTURE.md](https://github.com/Vision-Decision/Glance-X/blob/main/docs/ARCHITECTURE.md).

## Results

All results are on the full splits. Each direction was evaluated separately with these weights.

**Original benchmarks** (human labels). RefCOCO family, Acc@0.5 (%):

| | Glance-4B-preview<br><sub>(image-first only)</sub> | `image_first` | `text_first` | `joint` |
|---|:-:|:-:|:-:|:-:|
| RefCOCO val / testA / testB | 93.2 / 94.5 / 90.7 | 92.9 / 94.8 / 90.4 | 92.1 / 93.9 / 89.5 | 93.1 / 94.9 / 90.5 |
| RefCOCO+ val / testA / testB | 89.7 / 93.3 / 84.6 | 89.2 / 93.0 / 84.7 | 87.6 / 91.6 / 82.0 | 89.6 / 93.1 / 84.9 |
| RefCOCOg val / test | 90.0 / 90.4 | 90.0 / 90.5 | 89.3 / 89.0 | 90.3 / 90.5 |
| **8-split mean** | 90.8 | 90.7 | 89.4 | **90.9** |

gRefCOCO val, %:

| Metric | `image_first` | `text_first` | `joint` |
|---|:-:|:-:|:-:|
| AUROC of P(present) | 98.1 | 97.3 | **98.2** |
| multi-target F1 | 77.3 | 78.1 | **82.7** |

**Glance-Data eval set** (300 held-out images, 23,586 pipeline-labelled requests; in-distribution for this model):

| Metric | `image_first` | `text_first` | `joint` |
|---|:-:|:-:|:-:|
| Acc@0.5, English / Chinese | 88.1 / 86.8 | 87.2 / 85.9 | 87.8 / 86.8 |
| look-alike negatives rejected (attribute / relation) | 94.9 / 94.2 | 93.8 / 94.2 | **95.0 / 94.4** |
| multi-target F1 | 62.6 | 66.7 | **69.7** |

Full tables, including a stronger internal single-direction model that this checkpoint still trails by 1.5–3.8
points on RefCOCO, are in
[RESULTS.md](https://github.com/Vision-Decision/Glance-X/blob/main/docs/RESULTS.md).

## Training

- **Data.** 355k original requests (RefCOCO / RefCOCO+ / RefCOCOg, plus gRefCOCO negatives) and 1.83M requests from
  [Glance-Data](https://github.com/Vision-Decision/Glance-Data)
  ([🤗 dataset](https://huggingface.co/datasets/Vision-Decision/Glance-Data)). Glance-Data adds:
  - short, long, casual and Chinese requests
  - category-only and multi-target requests
  - look-alike and count, position and near-category negatives
- **Directions.** Each batch uses one direction: `image_first` 50%, `text_first` 25%, `joint` 25%.
- **Schedule.** 2 epochs = 6,100 steps at 8 images per step, LR 2e-4 (LoRA) and 5e-4 (heads), cosine schedule,
  bf16 with fp32 LoRA. This took about 3 h on one H200.

The training recipe is in [TRAINING.md](https://github.com/Vision-Decision/Glance-X/blob/main/docs/TRAINING.md).

## Limitations and license

- The model was trained on natural photos (COCO). It was not trained on screenshots or documents.
- The weights are licensed **CC BY-NC-SA 4.0**, for **non-commercial use** only, because the training data
  (gRefCOCO, Glance-Data) is non-commercial. The code on GitHub is Apache-2.0. The base model has its own license.

## Citation

```bibtex
@software{glancex2026,
  title  = {Glance-X: one model, three ways to look},
  author = {Vision-Decision},
  year   = {2026},
  url    = {https://github.com/Vision-Decision/Glance-X}
}
```
