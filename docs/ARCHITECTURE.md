# Glance-X architecture

Base: **Qwen3.5-4B-Base**, 32 layers (24 Gated DeltaNet + 8 gated full-attention, in a 3:1 pattern), with LoRA r=64
on every language-model linear layer and the vision tower frozen. The single-box recipe follows
[Glance](https://github.com/Vision-Decision/Glance): a presence gate, patch selection, four edge distributions, an
IoU head, explicit and hard negatives, and many requests answered in one forward pass.

## 1. Goal

One set of weights and one set of outputs, with three directions that the caller picks for each call:

| Direction | Shared (computed once, cacheable) | Computed per (image, request) pair | Best for |
|---|---|---|---|
| `image_first` (`i2t`) | the image | the request block | one image, many requests |
| `text_first` (`t2i`) | the request | the image block | one request, many images or frames; the image is read already knowing what to look for |
| `joint` (`bi`) | — | image + request | the most accurate single answer |

## 2. Token layout

Every (image, request) pair is made of three segments. **Their positions are fixed and do not depend on the
direction.**

```
image stream  I : <vision_start> <image_pad>*n <vision_end>    M-RoPE (t,h,w): 0 | (1, 1+i, 1+j) | 1+max(gh,gw)
text stream   T : <object_ref_start> expr <object_ref_end>     1-D, starting at the fixed offset text_pos0 = 66
readout       R : <box_start>                                  right after the text
```

`text_pos0` must be larger than 1 + the image grid's longer side, so the two streams never share a position. The
collate function asserts this. At 640 visual tokens, every COCO image is well below the limit.

## 3. Two stages

The 32 layers are split into two stages:

1. **Modality stage (layers 1..K).** I and T run on their own, each causal, and neither can see the other. Both can
   be computed once and cached. The text stream starts at a fixed offset and only attends within itself in this
   stage. RoPE depends only on relative positions, so cached text features do not depend on which image they are
   later paired with. `scripts/test_equivalence.py` checks that they are invariant to a position shift.
2. **Fusion stage (layers K+1..32).** The direction decides which stream comes first:
   - The **prefix** stream runs on its own (Gated DeltaNet with its own state; attention sees only the prefix), and
     every pair shares it.
   - Each **block** (the other stream + R) continues from the prefix. Gated DeltaNet layers start from the prefix's
     final recurrent state and its last (k−1) conv inputs. Full-attention layers see every prefix token plus the
     block, with bidirectional attention inside the block.
   - `image_first`: prefix = I, block = [T, R]. `text_first`: prefix = T without R, block = [I, R].
     `joint`: no prefix, block = [I, T, R], every token visible to every other (Gated DeltaNet runs in I→T→R order).

Many blocks share one prefix in a single batched forward pass. `glancex/_layers.py` implements this for both layer
types without materialising one copy of the prefix per block.

The heads always read R, which has seen both the image and the request in every direction, and the last-layer image
tokens. In `image_first` those are the shared image prefix. In `text_first` and `joint` each pair has its own
request-conditioned image tokens.

**Degenerate cases.** With K=0 and a causal block mask, `image_first` is exactly a standard image-prefix model and
`text_first` is exactly a text-prefix model. `scripts/test_equivalence.py` checks this against a plain Hugging Face
forward of the single sequence. K sets the trade-off between how deep the cacheable part goes and how deep fusion
goes. With the default K=8, the streams stay separate through the first two full-attention layers (layers 4 and 8).

## 4. Heads

The multi-box branch is added on top of the Glance single-box heads:

| Output | How | Supervision |
|---|---|---|
| presence | R → MLP → P(present) | every request, including hard negatives |
| centre (single box) | softmax(R · patch) | single target: the patches its box covers; multi-target: the union over all boxes |
| objectness (multi-box, new) | sigmoid(R · patch) per patch | positive if the patch centre lies in the central 0.5 of any target box (plus the nearest patch); focal loss; all zeros for negatives |
| edges | a 64-bin distribution for each of the four edges, read at the chosen patch | one random target per request and a patch it covers; Cramér + GIoU |
| IoU | P(IoU ≥ 0.5) | predicted box vs the best-matching target |

Decoding works as follows:

- **Single box:** argmax of the centre softmax.
- **Multi-box:** every patch with objectness > τ (default 0.5, at most 20) gives one box, followed by NMS at 0.5.
- **Confidence:** P(present) × P(IoU ≥ 0.5).

## 5. Training

- **Data.** The original training set (355k requests: RefCOCO / RefCOCO+ / RefCOCOg + gRefCOCO no-target, plus
  swapped negatives) is combined with the Glance-Data augmentation (1.83M requests). The augmentation adds:
  - short, medium, long and casual descriptions, in English and Chinese
  - category-only and every-instance requests
  - multi-target and function requests
  - same-category hard negatives, plus count, position and near-category negatives

  The two sets are merged per image. Each step samples up to 48 requests per image, with at least 1/3 from the
  original data.
- **Directions.** Each batch uses one direction: `image_first` 50%, `text_first` 25%, `joint` 25%. `image_first`
  batches use 24 requests per image; `text_first` and `joint` batches use 3 pairs per image.
- **Fixed shapes.** Image 642 tokens, text 44, `text_first` block 644, `joint` block 688. fla's Hopper kernels compile
  once per shape.
- **Optimiser.** AdamW, LR 2e-4 for LoRA and 5e-4 for the heads, 100 warm-up steps, then cosine to 5%. 2 epochs =
  6,100 steps at 8 images per step, about 1.8 s/step on one H200, 22 GB peak.
