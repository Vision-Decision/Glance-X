"""Locate stages where fp16 arithmetic affects Glance-X box geometry."""

import argparse
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from glancex import GlanceX
from glancex._mac_layers import Linear, Vision
from scripts.check_mac import gate_requests, read_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--min-error", type=float, default=2.0)
    p.add_argument("--variant", action="append")
    args = p.parse_args()
    ref = {(tuple(a["key"]), a["mode"]): a for a in read_rows(args.reference)}
    cases = defaultdict(list)
    for b in read_rows(args.candidate):
        mode = b["mode"]
        if mode == "cached":
            continue
        a = ref[tuple(b["key"]), mode]
        if a["present_prob"] > 0.5 and max(abs(x-y) for x, y in zip(a["box"], b["box"])) > args.min_error:
            cases[mode].append(a)
    file_of = {tuple(row["key"]): args.gate / "images" / f for f, rows in gate_requests(args.gate).items() for row in rows}
    from glancex import _mac
    with patch.object(_mac, "Vision", lambda module, proc: Vision(module, proc, precise=False)):
        model = GlanceX.from_pretrained(args.ckpt, args.base, device="mps")
    vlm = model._mac
    vision = {vlm.vision.patch, vlm.vision.m_up, vlm.vision.m_down}
    vision.update(v for block in vlm.vision.blocks for v in block if isinstance(v, Linear))
    lm = {v for layer in vlm.layers for v in (layer.mix.in_proj, layer.mix.out_proj, layer.up, layer.down)}
    delta = {layer.mix.in_proj for layer in vlm.layers if layer.gdn}
    original = Linear.__call__
    attention = {v for block in vlm.vision.blocks for v in block[1:3]}
    mlp = {v for block in vlm.vision.blocks for v in block[5:]}

    mx.disable_compile()
    for label, use_float in (
        ("attention", attention), ("vision_mlp", mlp),
        ("half", set()), ("vision32", vision), ("delta_in32", delta), ("lm32", lm), ("all32", vision | lm),
    ):
        if args.variant and label not in args.variant:
            continue

        def project(self, x):
            if self not in use_float:
                return original(self, x)
            f = x.astype(mx.float32).reshape(-1, x.shape[-1])
            w = self.w.astype(mx.float32)
            y = f @ w if self.bias is None else mx.addmm(self.bias.astype(mx.float32), f, w)
            return y.reshape(*x.shape[:-1], self.w.shape[-1])

        with patch.object(Linear, "__call__", project):
            for mode, rows in cases.items():
                errors, elapsed = [], []
                for a in rows:
                    start = perf_counter()
                    result, diagnostic = vlm.ask(
                        str(file_of[tuple(a["key"])]), [a["request"]], model._check_mode(mode), 0.5, diagnostics=True,
                    )[0]
                    errors.append(round(max(abs(x-y) for x, y in zip(a["box"], result.box)), 4))
                    elapsed.append((perf_counter() - start) * 1000)
                    print(label, mode, a["key"], errors[-1], diagnostic["center_indices"], flush=True)
                print(label, mode, "errors", errors, "median_ms", round(np.median(elapsed), 1), flush=True)


if __name__ == "__main__":
    main()
