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
