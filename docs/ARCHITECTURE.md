# Glance-X-Lite architecture

Base: **Qwen3.5-0.8B-Base**. It has 24 language-model layers (Gated DeltaNet and gated full attention, 3:1) and a
12-layer vision tower. LoRA r=64 sits on every language-model linear layer; the fast inference path folds it into
the base weights. The vision tower is frozen.

## Token layout

Every (image, request) pair is made of three segments. Their positions are fixed and do not depend on the
direction:

```
image stream  I : <vision_start> <image_pad>*n <vision_end>    M-RoPE (t,h,w): 0 | (1, 1+i, 1+j) | 1+max(gh,gw)
text stream   T : <object_ref_start> request <object_ref_end>  1-D, from the fixed offset text_pos0 = 66
readout       R : <box_start>                                   right after the request
```

## Two stages, two directions

1. **Modality stage (layers 1–2).** I and T each run alone, with causal attention, and do not see each other.
2. **Fusion stage (layers 3–24).** One stream is a *prefix* and the other a *block* that continues it. Many blocks
   share one prefix in a single forward pass. Gated DeltaNet layers start each block from the prefix's final
   recurrent state; attention layers let each block attend to the prefix and, in both directions, to itself.
   - `image_first`: prefix = I, block = [T, R].
   - `text_first`: prefix = T without R, block = [I, R]. The image tokens read the request.

The heads read R and the last-layer image tokens:
- presence gate
- patch-centre softmax plus a 64-bin distribution for each of the four edges (one box)
- IoU head
- per-patch objectness, which gives one box per target after NMS

Confidence is P(present) × P(IoU ≥ 0.5).

## Image cache (`image_first`)

In `image_first`, the image never reads a request, so every layer's image state can be computed once:

- `GlanceX.encode(images)` keeps, per image:
  - the Gated DeltaNet state and conv tail of each fusion layer
  - the attention keys and values
  - the final image tokens

  This is about 25 MB per image.
- `search(request, bank)` then runs only the request's own block against each cached image.
- `scripts/test_equivalence.py` checks the cached path against the full forward pass.

`text_first` cannot be cached this way, because there the image tokens depend on the request.

## Fast path

`fast=True` is the default on CUDA:
- LoRA folded into the bf16 weights
- images decoded and resized on worker threads, then normalised on the GPU
- masked attention through the efficient SDPA kernel
- vectorised multi-box NMS
- the language model and heads replayed as CUDA graphs, one per shape bucket

`devices=[...]` puts one replica on each GPU and splits batches across them. `fast=False` is the exact
training-time computation.
