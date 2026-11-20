<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.svg">
    <img src="assets/banner-light.svg" width="760" alt="Glance-X-Lite: find any object from a text request, both ways">
  </picture>
</p>

<p align="center"><b>English</b> · <a href="README.zh-CN.md">简体中文</a></p>

<p align="center">
  <a href="https://github.com/Vision-Decision/Glance-X/actions/workflows/ci.yml"><img src="https://github.com/Vision-Decision/Glance-X/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.4+">
  <a href="https://huggingface.co/Qwen/Qwen3.5-0.8B-Base"><img src="https://img.shields.io/badge/base-Qwen3.5--0.8B--Base-7C3AED" alt="Base model: Qwen3.5-0.8B-Base"></a>
  <a href="https://huggingface.co/Vision-Decision/Glance-X-Lite"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20model-Glance--X--Lite-FFD21E?logo=huggingface&logoColor=000" alt="Model on Hugging Face: Glance-X-Lite"></a>
  <a href="https://github.com/Vision-Decision/Glanece-Training"><img src="https://img.shields.io/badge/training-Glanece--Training-181717?logo=github&logoColor=white" alt="Training code: Glanece-Training"></a>
  <a href="https://github.com/Vision-Decision/Glance-Data"><img src="https://img.shields.io/badge/data-Glance--Data-181717?logo=github&logoColor=white" alt="Glance-Data (GitHub)"></a>
  <a href="https://huggingface.co/datasets/Vision-Decision/Glance-COCO"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20dataset-Glance--COCO-FFD21E?logo=huggingface&logoColor=000" alt="Glance-COCO (Hugging Face)"></a>
  <img src="https://img.shields.io/badge/weights-CC%20BY--NC--SA%204.0-6B7280" alt="Weights: CC BY-NC-SA 4.0">
  <a href="LICENSE"><img src="https://img.shields.io/badge/code-Apache%202.0-2F80ED?logo=apache&logoColor=white" alt="Code license: Apache-2.0"></a>
</p>

<p align="center">
  <a href="#two-directions">Two directions</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#results">Results</a> ·
  <a href="#speed">Speed</a> ·
  <a href="docs/ARCHITECTURE.md">Architecture</a> ·
  <a href="docs/RESULTS.md">Full results</a>
</p>

---

**Glance-X-Lite** finds objects in an image from a text request, such as *"the kid with glasses in the middle"* or
*"左边举起球拍的男孩"*. It answers with a **box** around every matching object, or says **the object is not in the
image**. It is a 0.8B model, small enough for a laptop.

## Two directions

One set of weights covers both common workloads, and you pick the direction for each call:

| Direction | What is cached | Best for |
|---|---|---|
| `image_first` &nbsp;image → text | **the image**: encode it once, then each request costs only its own ~30 tokens | one image, many requests; a photo library searched again and again |
| `text_first` &nbsp;text → image | the request; each new image reads it while it is encoded | one request over a stream of new images, e.g. camera or video frames; best multi-target recall |

<p align="center">
  <img src="assets/demo.gif" width="860" alt="Demo: 256 images, one request, a scan picks the right image, the view zooms in, and new requests find each target in it">
  <br>
  <sub>One request over 256 cached images (68 ms on one GPU) picks the image; then new requests find their targets
  inside it. Every box, score, ranking and timing is a real Glance-X-Lite output on held-out images.
  <a href="assets/demo.mp4">MP4 version</a> · made with <a href="demo/render_story.py"><code>demo/</code></a> ·
  images from MS COCO.</sub>
</p>

## How it works

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/architecture-dark.svg">
    <img src="assets/architecture-light.svg" width="900" alt="Glance-X-Lite architecture: a modality stage where image and text run separately, a fusion stage with two layouts, and four heads">
  </picture>
</p>

Glance-X-Lite is built on **Qwen3.5-0.8B-Base**: 24 layers (Gated DeltaNet and gated full attention), with LoRA on
every language-model linear layer and the vision tower frozen. The image tokens go **inside** the language model and
are fused with the request over 22 layers.

1. **Modality stage** (layers 1–2). Image and request run separately.
2. **Fusion stage** (layers 3–24). One stream is a shared prefix and the other a block that reads it:
   - `image_first`: prefix = image, block = [request, R]. The image never reads a request, so its state can be
     cached.
   - `text_first`: prefix = request, block = [image, R]. The image reads the request.
3. **Heads** read the readout token R and the image tokens:
   - presence: P(the request matches something in the image)
   - box: a patch centre plus a 64-bin distribution for each of the four edges
   - IoU: P(IoU ≥ 0.5)
   - multi-box: per-patch objectness, which yields one box for every target

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The training code and every experiment behind this design,
including the ones that did not work, are in [Glanece-Training](https://github.com/Vision-Decision/Glanece-Training).

## Quick start

```bash
git clone https://github.com/Vision-Decision/Glance-X && cd Glance-X
pip install -e ".[fla]"        # drop [fla] on machines without a CUDA GPU
```

```python
from glancex import GlanceX

m = GlanceX.from_pretrained("Vision-Decision/Glance-X-Lite")   # LoRA + heads (188 MB) + Qwen/Qwen3.5-0.8B-Base

# one image, many requests: the image is computed once
m.ask("tennis.jpg", ["the kid with glasses in the middle", "right racket", "左边举起球拍的男孩",
                     "boy on the left raising a blue racket", "a frisbee"])
# Result('the kid with glasses in the middle', present_prob=1.000, box=[250, 153, 369, 422], confidence=0.991, n_boxes=1)
# Result('right racket', present_prob=1.000, box=[534, 74, 634, 243], confidence=0.993, n_boxes=1)
# Result('左边举起球拍的男孩', present_prob=0.983, box=[62, 146, 260, 427], confidence=0.979, n_boxes=1)
# Result('boy on the left raising a blue racket', present_prob=0.012, box=[59, 141, 251, 427], confidence=0.011, n_boxes=0)
# Result('a frisbee', present_prob=0.000, box=[511, 134, 588, 261], confidence=0.000, n_boxes=0)

# every target, not just one
m.find("giraffes.jpg", "the three giraffes in the field").boxes
# [[516, 92, 634, 308], [50, 66, 250, 324], [330, 71, 516, 316]]

# one request over images you have already encoded (image cache, 25 MB per image)
bank = m.encode(photos)
m.search("a giraffe eating from someone's hand", bank)[0]
# Result("a giraffe eating from someone's hand", present_prob=0.998, box=[237, 39, 604, 335], confidence=0.381, n_boxes=0)

# one request over new images: each image reads the request while it is encoded
m.search("a giraffe eating from someone's hand", photos, mode="text_first")[0]
# Result("a giraffe eating from someone's hand", present_prob=0.948, box=[320, 51, 611, 320], confidence=0.456, n_boxes=0)
```

These outputs are real (CUDA, bf16, the default fast path).

| `Result` field | Meaning |
|---|---|
| `present_prob` | P(the request matches something in the image). Use it to reject requests. |
| `box` | the most likely target, `[x1, y1, x2, y2]` in pixels of the input image (filled in even when absent; gate on `present_prob`) |
| `iou_prob`, `confidence` | P(IoU ≥ 0.5), and `present_prob × iou_prob` as one score to threshold or rank on |
| `boxes`, `box_scores` | every target found by the multi-box head, with its objectness score |
| `image_index` | which input image the result belongs to (for `search`) |

Apple-silicon Macs are the target platform. On them, `device="mps"` uses the exact path (fp32, without `[fla]`).
The MPS path and its latency have not been tested yet.

## Results

Full splits, Acc@0.5 (%). Each direction is evaluated separately with the same weights. Larger Glance-X models (4B)
are shown for reference.

| Split | **Lite `image_first`** | **Lite `text_first`** | Glance-X2 (4B) | Glance-X `joint` (4B) |
|---|:-:|:-:|:-:|:-:|
| RefCOCO val / testA / testB | 90.8 / 92.7 / 87.8 | 89.9 / 91.8 / 86.1 | 93.6 / 95.1 / 90.5 | 93.1 / 94.9 / 90.5 |
| RefCOCO+ val / testA / testB | 85.3 / 90.2 / 80.2 | 83.1 / 88.0 / 76.5 | 90.4 / 93.4 / 85.5 | 89.6 / 93.1 / 84.9 |
| RefCOCOg val / test | 87.5 / 87.7 | 85.9 / 85.5 | 90.8 / 91.1 | 90.3 / 90.5 |
| **8-split mean** | **87.8** | **85.9** | 91.3 | 90.9 |

gRefCOCO, which has expressions with no target, one target or several targets (%):

| Split / metric | Lite `image_first` | Lite `text_first` |
|---|:-:|:-:|
| val: AUROC of P(present) / no-target rejected | 97.4 / 74.0 | 96.4 / 70.1 |
| val: multi-target F1 / ≥1 target hit | 75.0 / 78.8 | **76.1 / 85.3** |
| testA: multi-target F1 | 71.1 | **71.8** |
| testB: multi-target F1 | 59.8 | **62.2** |

Glance-Data eval set: 300 held-out images and 23,586 requests, covering short, long, casual and Chinese requests,
look-alike negatives and multi-target requests. It has the same distribution as part of the training data.

| Metric | Lite `image_first` | Lite `text_first` |
|---|:-:|:-:|
| Acc@0.5 (single target) | 86.4 | 85.8 |
| AUROC of P(present) | 99.7 | 99.5 |
| look-alike rejected: wrong attribute / wrong relation | 93.2 / 92.5 | 90.0 / 88.9 |
| multi-target F1 / group ≥1 hit | 59.6 / 76.7 | **64.6 / 82.6** |

What this shows:

- **Lite `image_first` is 3.5 points below Glance-X2 (4B) on the RefCOCO-family mean** (87.8 vs 91.3), at a fifth
  of the size. `text_first` is 5.4 points below it.
- **The two directions trade places.** `image_first` is slightly better on single targets and on rejection.
  `text_first` finds more of a group, because the image reads the request.

Reproduce with `python -m glancex.evaluate --ckpt Vision-Decision/Glance-X-Lite --mode image_first ...` (see
[docs/RESULTS.md](docs/RESULTS.md)).

## Speed

One H200, end to end (ms, N = 1 / 16 / 64 / 128). Same inputs as our comparison with the TypeSafe Jev API:
[`scripts/benchmark_twoway.py`](scripts/benchmark_twoway.py), [`scripts/compare_jev.py`](scripts/compare_jev.py).

| Workload | **Glance-X-Lite** | Jev 1.14 (API) |
|---|---|---|
| one image, N requests (`image_first`) | **36 / 44 / 64 / 94** | 90 / 97 / 109 / 122 |
| one request, N cached images (`image_first` + cache) | **34 / 33 / 36 / 42** | — |
| one request, N new images, encode + search (`image_first`) | 84 / 187 / 609 / 1,143 | 74 / 391 / 1,453 / 2,042 |
| one request, N new images (`text_first`) | 38 / 181 / 658 / 1,293 | (same as above) |

The Jev times include the network round trip from our server; Jev answers N new images with N concurrent requests.
Mac numbers will be added once measured.

## Limitations

- Natural photos only. The model was trained on COCO, not on screenshots or documents.
- The image cache takes about 25 MB of memory per image.
- `joint` is not available: Lite was trained in the two directions above only, and the API raises an error for
  `joint`.
- The weights are CC BY-NC-SA 4.0, for **non-commercial use** only, because the training data (gRefCOCO,
  Glance-Data) is non-commercial. The code is Apache-2.0.

## Citation

```bibtex
@software{glancex2026,
  title  = {Glance-X: find any object from a text request, both ways},
  author = {Vision-Decision},
  year   = {2026},
  url    = {https://github.com/Vision-Decision/Glance-X}
}
```

## Acknowledgements

- [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base): the base model.
- [Glance](https://github.com/Vision-Decision/Glance): the box recipe.
- [Glance-Data](https://github.com/Vision-Decision/Glance-Data): the new request types.
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention): Gated DeltaNet kernels.
- Datasets: RefCOCO / RefCOCO+ / RefCOCOg, gRefCOCO and MS COCO.
