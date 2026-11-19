# Training and evaluation

## Requirements

- One CUDA GPU. The reference run used one H200, peaked at 22 GB allocated, and took about 3 h.
- Python ≥ 3.10, PyTorch ≥ 2.4, transformers with Qwen3.5 support (≥ 5.9), and peft.
- `flash-linear-attention` for the Gated DeltaNet kernels. Without it, the PyTorch reference kernel from transformers
  is used, which works but is much slower.

```bash
pip install -e ".[fla]"
```

## Data layout

```
data/
  ann/                                    raw annotations
    refcoco/ refcocoplus/ refcocog/       {train,validation,test,testB}.parquet   (Hugging Face jxu124/*)
    grefcoco/grefs_unc.json               (FudanCVL/gRefCOCO)
    annotations/instances_train2014.json  (COCO 2014)
  images/coco_train2014/                  COCO train2014 images (all RefCOCO-family images are from this split)
  glance-data/                            from https://github.com/Vision-Decision/Glance-Data
    aug_train_all.jsonl                   1.83M augmented requests on the training images
    aug_val.jsonl                         the Glance-Data eval set (300 held-out images)
```

1. `scripts/prepare_refcoco.py` writes `data/prep/train.jsonl` and `data/prep/eval_<split>.jsonl`. Every image used by
   any eval split is removed from training.
2. `scripts/merge_data.py` merges the original records and Glance-Data into `data/train.jsonl`, one record per image.
   Original requests are marked `orig=True`, which is what `min_orig_frac` uses.

Record format (one JSON line per image):

```json
{"image_id": 324910, "file": "COCO_train2014_000000324910.jpg", "W": 640, "H": 425,
 "questions": [{"text": "man on the left", "present": true, "box": [0, 0, 347.8, 425], "boxes": [[0, 0, 347.8, 425]],
                "kind": "short", "lang": "en", "orig": false}]}
```

`box` is the single target (xyxy, pixels). `boxes` lists every target, and is empty or null for absent requests.

## Train

```bash
python -m glancex.train --cfg configs/glance-x-4b-k8.json
python -m glancex.train --cfg configs/smoke.json                       # 20 steps on 64 images, to check the setup
python -m glancex.train --cfg configs/glance-x-4b-k8.json --set lr=1e-4 out_dir=outputs/lr1e-4
```

Outputs go to `out_dir`:

- `config.json`
- `train_log.jsonl`: per-direction losses and train Acc@0.5, every `log_every` steps
- `ckpt_last.pt`: written every `save_every` steps
- `ckpt_final.pt`

A checkpoint holds only the LoRA weights, the heads and the config (about 0.5 GB). The base model is loaded from
`model_id`.

Resume with `--set resume=outputs/.../ckpt_last.pt`. To initialise from another checkpoint without its step
counter, use `init_ckpt`.

## Evaluate

```bash
python -m glancex.evaluate --ckpt outputs/glance-x-4b-k8/ckpt_final.pt --mode image_first \
    --sets refcoco_val,refcoco_testA,refcoco_testB,refcocoplus_val,refcocoplus_testA,refcocoplus_testB,refcocog_val,refcocog_test,grefcoco_val_merged,grefcoco_testA,grefcoco_testB \
    --prep_dir data/prep --glance_data data/glance-data/aug_val.jsonl --out results/k8_image_first.json
```

Run it once per `--mode`. `--limit_images N` gives a quick partial run. Results for the Glance-Data set are also
broken down `by_kind`, by kind and by kind:language.

## Config keys

All keys live in `glancex/config.py`. The ones that matter most:

| Key | Default | Meaning |
|---|---|---|
| `model_id` | `Qwen/Qwen3.5-4B-Base` | base model (Hugging Face id or local path) |
| `bidir_k` | 8 | layers in the modality stage (0 = fuse from the first layer) |
| `bidir_block_mask` | `block_bidir` | attention inside a fusion block; `causal` is only for the exactness test |
| `text_pos0` | 66 | fixed first position of the text stream |
| `lora_r` / `lora_alpha` | 64 / 128 | LoRA on every LLM linear layer |
| `mode_probs` | `0.5,0.25,0.25` | share of `image_first`, `text_first`, `joint` batches |
| `i2t_q_per_image` / `pair_q_per_image` | 24 / 3 | requests per image in each kind of batch |
| `min_orig_frac` | 0.34 | minimum share of original requests among those sampled per image |
| `max_vis_tokens` | 640 | visual tokens per image (one per 32×32 px) |
| `pad_*_to` | 642 / 44 / 644 / 688 | fixed sequence lengths during training; 0 = batch maximum (inference) |
| `w_obj`, `obj_shrink` | 1.0, 0.5 | multi-box loss weight; central share of a box whose patches count as positive |

## Exactness test

```bash
python scripts/test_equivalence.py --model_id Qwen/Qwen3.5-4B-Base    # CUDA; uses a tiny random config
```

It checks:

- each direction at K=0 against a Hugging Face forward of the single sequence
- that at K = all layers the request readout does not depend on the image
- isolation between requests that share a prefix
- that the text stream is invariant to a position shift

It prints `ALL_OK` when every check passes.
