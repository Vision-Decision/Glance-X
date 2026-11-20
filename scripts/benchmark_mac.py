"""Measure the M2 roofline workloads with synchronized warm medians.

Run from the checkout with python -m scripts.benchmark_mac. Model and images are supplied locally.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch
import warnings

from PIL import Image
import torch

from glancex import GlanceX
from glancex import _layers
from glancex.data import resize_dims

FLOORS = {
    374: dict(ask1=288.5, ask8=321.9, text1=288.9, encode=283.7, search1=11.2, search256=1222.3),
    638: dict(ask1=529.6, ask8=563.5, text1=530.1, encode=524.8, search1=11.3, search256=1237.8),
}
REQUEST = "the giraffe on the right"
WORKLOADS = tuple(FLOORS[374])


def elapsed(fn, sync):
    start = perf_counter()
    result = fn()
    sync()
    return result, (perf_counter() - start) * 1000


def measure(fn, sync, repeats, warmup):
    _, first = elapsed(fn, sync)
    warm = [elapsed(fn, sync)[1] for _ in range(warmup)]
    samples = [elapsed(fn, sync)[1] for _ in range(repeats)]
    return dict(first_ms=first, warmup_ms=warm, median_ms=median(samples), samples_ms=samples)


def profile_reference(model, image, mode, repeats):
    """Synchronize between operators so the elapsed times can be attributed to stages."""
    sync = torch.mps.synchronize
    totals = Counter()
    phase = "lm"

    def wrap(fn, name):
        def timed(*a, **kw):
            nonlocal phase
            previous = phase
            if name in ("vision", "single head", "multi head"):
                phase = name
            label = f"{phase} SDPA" if name == "SDPA" else name
            value, ms = elapsed(lambda: fn(*a, **kw), sync)
            totals[label] += ms
            phase = previous
            return value
        return timed

    def count_norm(fn, name):
        # Record GEMMs by their current stage, which is set before the vision tower.
        def timed(*a, **kw):
            return fn(*a, **kw) if phase.endswith(" head") else wrap(fn, f"{phase} {name}")(*a, **kw)
        return timed

    methods = [
        (model, "_item", "preprocess"),
        (model.net, "image_features", "vision"),
        (model.heads, "forward", "single head"),
        (model.heads, "decode_multi", "multi head"),
        (_layers, "_gdr", "LM delta rule"),
        (torch.nn.functional, "scaled_dot_product_attention", "SDPA"),
    ]
    model.find(image, REQUEST, mode=mode)
    sample = []
    with ExitStack() as stack:
        for obj, attr, label in methods:
            stack.enter_context(patch.object(obj, attr, wrap(getattr(obj, attr), label)))
        stack.enter_context(patch.object(torch.nn.functional, "linear", count_norm(torch.nn.functional.linear, "linear")))
        for _ in range(repeats):
            totals.clear()
            _, total = elapsed(lambda: model.find(image, REQUEST, mode=mode), sync)
            parts = dict(totals)
            parts["vision other"] = parts.pop("vision") - parts["vision SDPA"] - parts["vision linear"]
            parts["total"] = total
            parts["remainder"] = total - sum(v for k, v in parts.items() if k != "total")
            sample.append(parts)
    return {k: median(s[k] for s in sample) for k in sample[0]}


def profile_mlx_operators(model, image, repeats):
    import mlx.core as mx
    from glancex import _mac_layers

    counts = Counter()
    scope = "LM"

    def timed(fn, label):
        def call(*args, **kwargs):
            mx.eval(args, kwargs)
            start = perf_counter()
            out = fn(*args, **kwargs)
            mx.eval(out)
            counts[f"{scope} {label}"] += (perf_counter() - start) * 1000
            return out
        return call

    def vision(fn):
        def call(*args):
            nonlocal scope
            scope = "vision"
            try:
                return fn(*args)
            finally:
                scope = "LM"
        return call

    tracked = [
        (_mac_layers.Linear, "__call__", "projection"),
        (mx.fast, "scaled_dot_product_attention", "SDPA"),
        (_mac_layers, "delta_rule", "delta"),
        (mx, "conv1d", "convolution"),
    ]
    model.find(image, REQUEST)
    samples = []
    with ExitStack() as stack:
        stack.enter_context(patch.object(_mac_layers.Vision, "__call__", vision(_mac_layers.Vision.__call__)))
        for obj, attr, label in tracked:
            stack.enter_context(patch.object(obj, attr, timed(getattr(obj, attr), label)))
        for _ in range(repeats):
            counts.clear()
            _, ms = elapsed(lambda: model.find(image, REQUEST), mx.synchronize)
            samples.append(dict(counts, total=ms))
    return {k: median(s[k] for s in samples) for k in samples[0]}


def probe_mlx(model, tokens, runs, warmup):
    """Calibrate layout-sensitive GEMMs and streaming bandwidth on this machine."""
    import mlx.core as mx

    rows = {}
    vlm = model._mac
    for label, linear, m in [
        ("vision QKV", vlm.vision.blocks[0][1], 4 * tokens),
        ("vision MLP up", vlm.vision.blocks[0][5], 4 * tokens),
        ("vision MLP down", vlm.vision.blocks[0][6], 4 * tokens),
        ("LM GDN in", vlm.layers[0].mix.in_proj, tokens + 2),
        ("LM MLP up", vlm.layers[0].up, tokens + 2),
        ("LM MLP down", vlm.layers[0].down, tokens + 2),
        ("cache LM MLP up", vlm.layers[0].up, len(model.tok.text(REQUEST))),
    ]:
        k, n = linear.w.shape
        x = mx.random.normal((m, k)).astype(mx.float16)
        w, contiguous = linear.w, mx.contiguous(linear.w)
        mx.eval(x, w, contiguous)
        rows[label] = {}
        for name, weight in (("transposed", w), ("contiguous", contiguous)):
            result = measure(lambda: mx.eval(x @ weight), mx.synchronize, runs, warmup)
            result["tflops"] = 2 * m * k * n / result["median_ms"] / 1e9
            result["shape"] = [m, k, n]
            rows[label][name] = result
    x = mx.zeros((256 * 1024 * 1024,), dtype=mx.float16)
    mx.eval(x)
    speed = measure(lambda: mx.eval(x + 1), mx.synchronize, runs, warmup)
    speed["read_write_gb_s"] = 2 * x.nbytes / speed["median_ms"] / 1e6
    return dict(projections=rows, bandwidth=speed)


def image_tokens(model, path):
    with Image.open(path) as img:
        w, h = resize_dims(*img.size, model.cfg.max_vis_tokens, model.cfg.max_upscale)
    return w // 32 * (h // 32)


def run(model, args, sync):
    with args.gate.joinpath("eval_refcoco_val.jsonl").open() as f:
        records = [json.loads(line) for line in f]
    coco = [str(args.gate / "images" / r["file"]) for r in records if (r["W"], r["H"]) == (640, 480)]
    images = {374: str(args.demo), 638: coco[0]}
    data = []
    for n in args.tokens or (374, 638):
        image = images[n]
        assert image_tokens(model, image) == n
        if args.probe:
            probe = probe_mlx(model, n, args.runs, args.warmup)
            for label, layouts in probe["projections"].items():
                print(f"probe {n} {label}: " + ", ".join(
                    f"{layout} {info['median_ms']:.3f} ms / {info['tflops']:.2f} TF/s" for layout, info in layouts.items()
                ), flush=True)
            print(f"probe {n} bandwidth: {probe['bandwidth']['read_write_gb_s']:.1f} GB/s", flush=True)
            data.append(dict(engine=args.engine, tokens=n, probe=probe))
        if args.operators:
            ops = profile_mlx_operators(model, image, args.runs)
            print(f"operators {args.engine} N={n}: {json.dumps(ops)}", flush=True)
            data.append(dict(engine=args.engine, tokens=n, operators=ops))
        if args.profile:
            for mode in ("image_first", "text_first"):
                if args.engine != "mlx":
                    stages = profile_reference(model, image, mode, args.runs)
                else:
                    stages = model._mac.profile(image, REQUEST, mode, args.runs)
                print(f"profile {args.engine} N={n} {mode}: {json.dumps(stages)}", flush=True)
                data.append(dict(engine=args.engine, tokens=n, mode=mode, profile=stages))
        if args.profile_only or args.probe_only or args.operators_only:
            continue
        for work in args.workload or [*WORKLOADS[:-1], *(["search256"] if n == 638 else [])]:
            bank = None
            if work == "search1":
                bank = model.encode([image])
            elif work == "search256":
                candidates = [image] if n == 374 else coco[:args.bank_seeds]
                seed = model.encode(candidates, batch_size=1)
                # Repeating distinct real encoded rows gives a full-size, materialized bank.
                copies, extra = divmod(256, len(seed))
                bank = type(seed).concat([seed] * copies + ([seed.slice(0, extra)] if extra else []))
                del seed
            if work == "ask1":
                fn = lambda: model.find(image, REQUEST)
            elif work == "ask8":
                fn = lambda: model.ask(image, [REQUEST] * 8)
            elif work == "text1":
                fn = lambda: model.find(image, REQUEST, mode="text_first")
            elif work == "encode":
                fn = lambda: model.encode([image])
            else:
                if args.engine == "reference" and args.reference_search_chunk < 256:
                    fn = lambda: model._search_bank(REQUEST, bank, 0.5, chunk=args.reference_search_chunk)
                else:
                    fn = lambda: model.search(REQUEST, bank, sort=False)
            sync()
            row = measure(fn, sync, args.runs, args.warmup)
            floor = FLOORS[n][work]
            row.update(engine=args.engine, tokens=n, workload=work, floor_ms=floor, target_ms=floor / 0.85)
            data.append(row)
            ms = row["median_ms"]
            warm = ", ".join(f"{t:.1f}" for t in row["warmup_ms"])
            print(f"| {args.engine} | {n} | {work} | {ms:.2f} | {floor / 0.85:.2f} | {100 * ms / floor:.1f}% | "
                  f"{row['first_ms']:.2f} | {warm} |", flush=True)
            del fn
            bank = None
            if args.engine != "mlx":
                torch.mps.empty_cache()
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--base")
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--engine", choices=("reference", "mps16", "mlx"), default="mlx")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--tokens", type=int, choices=FLOORS, action="append")
    parser.add_argument("--workload", choices=WORKLOADS, action="append")
    parser.add_argument("--bank-seeds", type=int, default=8)
    parser.add_argument("--reference-search-chunk", type=int, default=256)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--operators", action="store_true", help="attribute MLX time to synchronized operators")
    parser.add_argument("--probe", action="store_true", help="measure native GEMM layouts and memory bandwidth")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--operators-only", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.runs < 10 or args.warmup < 1:
        parser.error("use at least 10 measured runs and one warm-up run")
    args.profile |= args.profile_only
    args.probe |= args.probe_only
    args.operators |= args.operators_only
    if args.engine != "mlx" and (args.probe or args.operators):
        parser.error("--probe and --operators require --engine mlx")
    warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available")
    if args.engine != "mlx":
        sync = torch.mps.synchronize
    else:
        import mlx.core as mx
        sync = mx.synchronize
    def load():
        model = GlanceX.from_pretrained(args.ckpt, base=args.base, device="mps", fast=args.engine == "mlx")
        if args.engine == "mps16":
            from glancex.inference import merge_lora
            merge_lora(model.net.hf)
            model.net.half()
        if args.engine == "mlx" and model._mac is None:
            parser.error("MLX is required for --engine mlx")
        return model

    model, load_ms = elapsed(load, sync)
    print(f"{args.engine} model load {load_ms:.1f} ms, request block {len(model.tok.text(REQUEST))} tokens", flush=True)
    print("| Engine | N | Workload | Median (ms) | Target (ms) | % of floor | First (ms) | Warm-ups (ms) |", flush=True)
    print("|---|---:|---|---:|---:|---:|---:|---:|", flush=True)
    data = run(model, args, sync)
    if args.output:
        args.output.write_text(json.dumps(dict(load_ms=load_ms, data=data), indent=2) + "\n")


if __name__ == "__main__":
    main()
