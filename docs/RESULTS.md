# Results

Checkpoint: Glance-X-4B, K=8, step 6100 (end of training). Every number is on the **full** split. Each direction was
run separately with the same weights (CUDA, bf16, 640 visual tokens).

Two kinds of results are kept apart:

1. **Original benchmarks**, with human labels: RefCOCO / RefCOCO+ / RefCOCOg and gRefCOCO.
2. **Glance-Data eval set**: our pipeline-generated benchmark on 300 held-out images. It has the same distribution as
   part of the training data.

Reference rows:

- **Glance-4B-preview**: the public [Glance](https://github.com/Vision-Decision/Glance) model. Image-first only,
  fine-tuned from Qwen3.5-4B.
- **internal single-direction**: an unreleased image-first model with the same box heads. It was fine-tuned from the
  instruction-tuned Qwen3.5-4B on the original data only. It is the strongest single-direction model we have, and
  it shows the gap still to close.

## 1. Original benchmarks

### RefCOCO family, Acc@0.5 (%)

| Split | Glance-4B-preview | internal single-direction | Glance-X `image_first` | Glance-X `text_first` | Glance-X `joint` |
|---|:-:|:-:|:-:|:-:|:-:|
| RefCOCO val | 93.2 | 94.9 | 92.9 | 92.1 | 93.1 |
| RefCOCO testA | 94.5 | 96.1 | 94.8 | 93.9 | 94.9 |
| RefCOCO testB | 90.7 | 92.6 | 90.4 | 89.5 | 90.5 |
| RefCOCO+ val | 89.7 | 91.7 | 89.2 | 87.6 | 89.6 |
| RefCOCO+ testA | 93.3 | 94.8 | 93.0 | 91.6 | 93.1 |
| RefCOCO+ testB | 84.6 | 88.5 | 84.7 | 82.0 | 84.9 |
| RefCOCOg val | 90.0 | 92.2 | 90.0 | 89.3 | 90.3 |
| RefCOCOg test | 90.4 | 92.3 | 90.5 | 89.0 | 90.5 |
| **mean** | 90.8 | 92.9 | 90.7 | 89.4 | 90.9 |

### gRefCOCO (%)

`reject_rate` is the share of no-target expressions answered "not present" (P(present) < 0.5). `multi_f1` is the
greedy-matched F1 at IoU 0.5 between the multi-box output and all targets. `multi_hit_any` is the share of
multi-target expressions whose single box hits one of the targets.

| Split | Metric | internal single-direction | `image_first` | `text_first` | `joint` |
|---|---|:-:|:-:|:-:|:-:|
| val (merged) | reject_rate | 77.7 | 79.2 | 75.9 | 78.9 |
| val (merged) | AUROC of P(present) | 96.4 | 98.1 | 97.3 | 98.2 |
| val (merged) | Acc@0.5, single target | 94.9 | 92.9 | 92.1 | 93.0 |
| val (merged) | multi_hit_any | — | 81.2 | 87.5 | 89.0 |
| val (merged) | multi_f1 | — | 77.3 | 78.1 | 82.7 |
| testA | reject_rate | — | 70.5 | 65.6 | 71.3 |
| testA | AUROC | — | 97.3 | 96.2 | 97.4 |
| testA | Acc@0.5, single target | — | 92.6 | 91.8 | 92.7 |
| testA | multi_hit_any | — | 77.0 | 85.9 | 87.5 |
| testA | multi_f1 | — | 72.7 | 73.4 | 78.3 |
| testB | reject_rate | — | 73.1 | 69.7 | 73.0 |
| testB | AUROC | — | 93.4 | 92.7 | 93.5 |
| testB | Acc@0.5, single target | — | 83.4 | 82.3 | 83.7 |
| testB | multi_hit_any | — | 74.2 | 81.1 | 82.3 |
| testB | multi_f1 | — | 63.8 | 66.5 | 72.2 |

"val (merged)" is gRefCOCO val (no-target + multi-target) together with RefCOCO val (single target) on the same
images. This gRefCOCO release has no single-target val expressions. The internal model has no multi-box head.

## 2. Glance-Data eval set

There are 300 held-out RefCOCO-val images and 23,586 requests, and labels are about 90% precise on audit. **This set
has the same distribution as part of the Glance-X training data, and the internal single-direction model never saw
these request types.** Read the comparison as "what the new data teaches", not as a fair benchmark between the two
models.

| Kind | Metric | internal single-direction | `image_first` | `text_first` | `joint` |
|---|---|:-:|:-:|:-:|:-:|
| short | Acc@0.5 | 76.6 | 86.7 | 86.4 | 87.6 |
| medium | Acc@0.5 | 82.4 | 90.8 | 90.5 | 91.0 |
| long | Acc@0.5 | 84.6 | 91.3 | 92.0 | 92.0 |
| casual | Acc@0.5 | 80.1 | 88.5 | 88.2 | 89.2 |
| category name only | Acc@0.5 | 40.3 | 85.0 | 86.5 | 86.7 |
| look-alike: wrong attribute | reject_rate | 16.4 | 94.9 | 93.8 | 95.0 |
| look-alike: wrong relation | reject_rate | 8.1 | 94.2 | 94.2 | 94.4 |
| wrong count | reject_rate | 56.8 | 99.9 | 99.9 | 99.9 |
| wrong position | reject_rate | 87.5 | 99.0 | 98.2 | 99.0 |
| near category, absent | reject_rate | 91.2 | 99.9 | 100.0 | 99.9 |
| group of targets | multi_hit_any | 71.8 | 78.5 | 85.1 | 87.1 |
| every instance of a category | multi_hit_any | 36.5 | 61.4 | 65.0 | 66.8 |
| gRefCOCO-style multi | multi_hit_any | 80.0 | 80.9 | 86.4 | 86.6 |
| by function ("something to sit on") | multi_hit_any | 38.1 | 58.3 | 59.6 | 60.6 |

Overall:

| | `image_first` | `text_first` | `joint` |
|---|:-:|:-:|:-:|
| AUROC of P(present) | 99.82 | 99.71 | 99.82 |
| multi-target F1 | 62.6 | 66.7 | 69.7 |
| multi-target exact count | 39.1 | 43.0 | 46.7 |
| Acc@0.5 English / Chinese | 88.1 / 86.8 | 87.2 / 85.9 | 87.8 / 86.8 |
| eval wall time, 300 images (one shared H200) | 18 min | 63 min | 61 min |

`text_first` and `joint` are evaluated one pair per block at batch size 1, so their wall time is dominated by
per-call overhead, not by FLOPs.

## 3. K=8 vs K=0 (training metrics)

Training-batch metrics averaged over the last ~1,100 steps, with the same data and schedule:

| | K=8 | K=0 |
|---|:-:|:-:|
| `image_first` Acc@0.5 (train batches) | 87.0 | 87.4 |
| `text_first` Acc@0.5 | 87.4 | 86.8 |
| `joint` Acc@0.5 | 89.1 | 89.1 |
| presence loss i2t / t2i / joint | 0.043 / 0.052 / 0.043 | 0.039 / 0.052 / 0.039 |
| s / step, peak memory | 1.79 s, 22.3 GB | 1.94 s, 22.3 GB |

On training batches the two are within noise of each other. So K=8 gets an 8-layer cacheable modality stage at
little cost. The full benchmark evaluation was run for K=8 only.

## 4. Open questions

The gap to the internal single-direction model (1.5–3.8 points on RefCOCO) has three candidate causes that have not
been separated yet:

1. **Base vs instruct starting point.** Glance-X starts from Qwen3.5-4B-Base.
2. **Data mix.** About 2/3 of the sampled requests are new types, so the original data gets fewer updates.
3. **The direction mix and the K=8 split.** Only half the batches are `image_first`.

A K=0 benchmark run, and a run with a larger original-data share, would separate them.

## Glance-X2

Glance-X2 starts from the Glance-X K=8 checkpoint above and trains 1 more epoch (3,050 steps) in the
`image_first` direction only, on the same data, with LoRA LR 1e-4 and head LR 5e-4
(`configs/glance-x2-4b.json`). Two ablations add a light two-way decoder (SAM-style, width 512, 2 or 4 layers) whose
outputs are added onto the existing heads through zero-initialised layers (`configs/ablations/`). All three runs used
the same schedule, so the comparison isolates the decoder. Every number is from the full splits, evaluated in
`image_first`.

### Accuracy (%)

| Metric | Glance-X `image_first` | Glance-X `joint` | **Glance-X2** | + decoder (2 layers) | + decoder (4 layers) |
|---|:-:|:-:|:-:|:-:|:-:|
| RefCOCO val / testA / testB | 92.9 / 94.8 / 90.4 | 93.1 / 94.9 / 90.5 | 93.6 / 95.1 / 90.5 | 93.5 / 95.2 / 91.1 | 93.6 / 95.4 / 91.2 |
| RefCOCO+ val / testA / testB | 89.2 / 93.0 / 84.7 | 89.6 / 93.1 / 84.9 | 90.4 / 93.4 / 85.5 | 89.8 / 93.2 / 85.8 | 90.3 / 93.3 / 85.1 |
| RefCOCOg val / test | 90.0 / 90.5 | 90.3 / 90.5 | 90.8 / 91.1 | 91.0 / 91.0 | 91.0 / 90.8 |
| **RefCOCO family mean** | 90.7 | 90.9 | **91.3** | **91.3** | **91.3** |
| gRefCOCO val: reject_rate / AUROC | 79.2 / 98.1 | 78.9 / 98.2 | 78.6 / 98.3 | 77.1 / 98.2 | 77.2 / 98.2 |
| gRefCOCO val: multi-target F1 / hit-any | 77.3 / 81.2 | **82.7 / 89.0** | 80.6 / 85.7 | 79.8 / 85.1 | 79.6 / 84.9 |
| gRefCOCO testA: multi-target F1 | 72.7 | **78.3** | 76.1 | 75.5 | 74.1 |
| gRefCOCO testB: multi-target F1 | 63.8 | **72.2** | 68.4 | 67.8 | 66.5 |
| Glance-Data: multi-target F1 / exact count | 62.6 / 39.1 | **69.7 / 46.7** | 67.3 / 43.4 | 66.8 / 42.5 | 67.0 / 42.6 |
| Glance-Data: group, ≥1 target hit | 78.5 | **87.1** | 81.0 | 80.9 | 81.5 |
| Glance-Data: look-alike attribute / relation rejected | 94.9 / 94.2 | 95.0 / 94.4 | 94.6 / 94.7 | 95.2 / 94.7 | 94.9 / 95.2 |

What this shows:

- **One more epoch of `image_first` helps.** The RefCOCO mean rises from 90.7 to 91.3, above `joint`. The
  multi-target F1 on gRefCOCO val rises from 77.3 to 80.6, which closes about half of the gap to `joint`.
- **The two-way decoder adds nothing on top.** It ties on RefCOCO and is 0.3–2.0 F1 points lower on multi-target
  requests than the run without it. A small decoder (2–4 layers, width 512) on top of the cached image tokens does not
  reproduce what `joint` gets from letting the image tokens read the request through 24 full-size layers. Possible
  reasons, not tested: the small width, the single epoch, and zero-initialised residuals that start from a strong
  model.

### Latency (ms, one H200, end to end, N = 1 / 16 / 64 / 128)

Inputs as in `scripts/compare_jev.py`: one 640×480 Glance-Data image with its first N requests, and a short request
over the first N other images. `scripts/benchmark_twoway.py` measures the median of 5 runs after 2 warm-up runs.

| Workload | Glance-X2 | + decoder (2 layers) | + decoder (4 layers) |
|---|---|---|---|
| one image, N requests | 58 / 79 / 137 / 219 | 59 / 78 / 145 / 228 | 60 / 81 / 147 / 234 |
| one request, N cached images | 48 / 46 / 58 / 71 | 46 / 48 / 55 / 80 | 56 / 50 / 59 / 86 |
| one request, N new images (encode + search) | 123 / 455 / 1,656 / 3,236 | 118 / 453 / 1,658 / 3,248 | 118 / 460 / 1,661 / 3,258 |

The image cache takes 55 MB per image. For reference, the same inputs through the TypeSafe Jev 1.14 API:
- one image, N requests: 90 / 97 / 109 / 122 ms
- one request, N new images, as N concurrent requests: 74 / 391 / 1,453 / 2,042 ms

The original Glance-X code took 148 / 201 / 247 / 345 ms for one image with N requests, and 138 / 754 / 2,996 /
5,966 ms for one request over N new images (`text_first`).
