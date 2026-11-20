"""Measure the M2 roofline workloads with synchronized warm medians.

Run from the checkout with python -m scripts.benchmark_mac. Model and images are supplied locally.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from functools import wraps
import json
from pathlib import Path
from statistics import median
from time import perf_counter, sleep
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
REQUEST = "the giraffe on the right of the picture"
WORKLOADS = tuple(FLOORS[374])


def elapsed(fn, sync):
    start = perf_counter()
    result = fn()
    sync()
    return result, (perf_counter() - start) * 1000


def measure(fn, sync, repeats, warmup, trace=None, expected=None, spacing_ms=0):
    before_trace = trace() if trace else 0.0
    result, first = elapsed(fn, sync)
    if expected is not None and len(result) != expected:
        raise RuntimeError(f"Expected {expected} outputs, got {len(result)}")
    del result
    first_trace = trace() - before_trace if trace else None
    def repeated():
        if spacing_ms:
            sleep(spacing_ms / 1000)
        return elapsed(fn, sync)[1]

    warm = [repeated() for _ in range(warmup)]
    samples = [repeated() for _ in range(repeats)]
    data = dict(first_ms=first, warmup_ms=warm, median_ms=median(samples), samples_ms=samples)
    if first_trace is not None:
        data["first_graph_trace_ms"] = first_trace
    return data


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


def profile_mlx_operators(call, repeats):
    import mlx.core as mx
    from glancex import _mac_layers
    from glancex._mac import BoxDecoder

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

    def in_scope(fn, label):
        def invoke(*args, **kwargs):
            nonlocal scope
            previous = scope
            scope = label
            try:
                return fn(*args, **kwargs)
            finally:
                scope = previous
        return invoke

    tracked = [
        (_mac_layers.Linear, "__call__", "projection"),
        (mx.fast, "scaled_dot_product_attention", "SDPA"),
        (_mac_layers, "delta_rule", "delta"),
        (_mac_layers, "depthwise_silu", "convolution"),
        (mx.fast, "rms_norm", "RMS norm"),
        (mx.fast, "layer_norm", "LayerNorm"),
    ]
    call()
    samples = []
    with ExitStack() as stack:
        mx.disable_compile()
        stack.callback(mx.enable_compile)
        stack.enter_context(patch.object(_mac_layers.Vision, "__call__", in_scope(_mac_layers.Vision.__call__, "vision")))
        for name in ("prepare", "decode"):
            stack.enter_context(patch.object(BoxDecoder, name, in_scope(getattr(BoxDecoder, name), "heads")))
        for obj, attr, label in tracked:
            stack.enter_context(patch.object(obj, attr, timed(getattr(obj, attr), label)))
        for _ in range(repeats):
            counts.clear()
            _, ms = elapsed(call, mx.synchronize)
            samples.append(dict(counts, total=ms))
    return {k: median(s[k] for s in samples) for k in samples[0]}


def probe_mlx(model, tokens, runs, warmup):
    """Calibrate layout-sensitive GEMMs and streaming bandwidth on this machine."""
    import mlx.core as mx

    from glancex._mac_kernels import delta_rule
    from scripts._mac_projection_probe import skinny_matmul, skinny_mma

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
        ("wide LM GDN in", vlm.layers[0].mix.in_proj, 256 * len(model.tok.text(REQUEST))),
        ("wide LM MLP up", vlm.layers[0].up, 256 * len(model.tok.text(REQUEST))),
        ("wide LM MLP down", vlm.layers[0].down, 256 * len(model.tok.text(REQUEST))),
    ]:
        k, n = linear.w.shape
        x = mx.random.normal((m, k)).astype(mx.float16)
        w = mx.contiguous(linear.w.T).T
        contiguous = mx.contiguous(w)
        mx.eval(x, w, contiguous)
        rows[label] = {}
        for name, weight in (("transposed", w), ("contiguous", contiguous)):
            result = measure(lambda: mx.eval(x @ weight), mx.synchronize, runs, warmup)
            result["tflops"] = 2 * m * k * n / result["median_ms"] / 1e9
            result["shape"] = [m, k, n]
            rows[label][name] = result
        if n % 64:
            padded = mx.pad(w.T, [(0, (-n) % 64), (0, 0)]).T
            mx.eval(padded)
            result = measure(lambda: mx.eval((x @ padded)[:, :n]), mx.synchronize, runs, warmup)
            result["tflops"] = 2 * m * k * n / result["median_ms"] / 1e9
            rows[label]["pad N to 64"] = result
        if m % 32 and m >= 256:
            aligned = mx.pad(x, [(0, (-m) % 32), (0, 0)])
            mx.eval(aligned)
            result = measure(lambda: mx.eval((aligned @ w)[:m]), mx.synchronize, runs, warmup)
            result["tflops"] = 2 * m * k * n / result["median_ms"] / 1e9
            rows[label]["pad M to 32"] = result
        if label.startswith("vision"):
            xf, wf = x.astype(mx.float32), w.astype(mx.float32)
            mx.eval(xf, wf)
            result = measure(lambda: mx.eval(xf @ wf), mx.synchronize, runs, warmup)
            result["tflops"] = 2 * m * k * n / result["median_ms"] / 1e9
            rows[label]["fp32 arithmetic"] = result
        if m <= 16:
            for name, op in [
                ("swap M and N", lambda x, w: (w @ x.T).T),
                ("pad M to 16", lambda x, w: (mx.pad(x, [(0, 16 - m), (0, 0)]) @ w.T)[:m]),
                ("custom narrow", skinny_matmul),
                ("SIMD matrix 128", lambda x, w: skinny_mma(x, w, 128)),
                ("SIMD matrix 256", skinny_mma),
                ("SIMD matrix 512", lambda x, w: skinny_mma(x, w, 512)),
            ]:
                result = measure(lambda: mx.eval(op(x, w.T)), mx.synchronize, runs, warmup)
                result["tflops"] = 2 * m * k * n / result["median_ms"] / 1e9
                rows[label][name] = result
    delta = {}
    for batch, length in ((1, tokens + 2), (1, len(model.tok.text(REQUEST))), (256, len(model.tok.text(REQUEST)))):
        shape = (batch, length, 16, 128)
        q, k = (mx.random.normal(shape) / 128**0.5 for _ in range(2))
        v = mx.random.normal(shape).astype(mx.float16)
        decay, beta = mx.full(shape[:-1], 0.8), mx.full(shape[:-1], 0.7)
        state = None if length > 16 else mx.random.normal((batch, 16, 128, 128)) * 0.02
        mx.eval(q, k, v, decay, beta, state)
        delta[f"B={batch} L={length}"] = {}
        for width, vector in ((8, 1), (16, 1), (32, 1), (8, 4), (16, 4), (32, 4)):
            fn = lambda: mx.eval(delta_rule(q, k, v, decay, beta, state, save_state=state is None, width=width, vector=vector))
            delta[f"B={batch} L={length}"][f"{width} lanes ×{vector}"] = measure(fn, mx.synchronize, runs, warmup)
    q, k, v = (mx.random.normal((1, 4 * tokens, 12, 64)).astype(mx.float16).transpose(0, 2, 1, 3) for _ in range(3))
    mx.eval(q, k, v)
    attention = {}
    q32, k32, v32 = (x.astype(mx.float32) for x in (q, k, v))
    mx.eval(q32, k32, v32)
    for name, fn in [
        ("default", lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125)),
        ("forced fused", lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, force_fused=True)),
        ("separate", lambda: mx.softmax((q @ k.swapaxes(-1, -2)) * 0.125, axis=-1, precise=True) @ v),
        ("fp32 fused", lambda: mx.fast.scaled_dot_product_attention(q32, k32, v32, scale=0.125, force_fused=True)),
    ]:
        attention[name] = measure(lambda: mx.eval(fn()), mx.synchronize, runs, warmup)
    x = mx.zeros((256 * 1024 * 1024,), dtype=mx.float16)
    mx.eval(x)
    speed = measure(lambda: mx.eval(x + 1), mx.synchronize, runs, warmup)
    speed["read_write_gb_s"] = 2 * x.nbytes / speed["median_ms"] / 1e6
    return dict(projections=rows, bandwidth=speed, delta=delta, vision_sdpa=attention)


def image_tokens(model, path):
    with Image.open(path) as img:
        w, h = resize_dims(*img.size, model.cfg.max_vis_tokens, model.cfg.max_upscale)
    return w // 32 * (h // 32)


def run(model, args, sync, checkpoint, trace, baselines):
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
            for label, variants in {**probe["delta"], "vision SDPA": probe["vision_sdpa"]}.items():
                print(f"probe {n} {label}: " + ", ".join(
                    f"{variant} {info['median_ms']:.3f} ms (first {info['first_ms']:.2f})" for variant, info in variants.items()
                ), flush=True)
            print(f"probe {n} bandwidth: {probe['bandwidth']['read_write_gb_s']:.1f} GB/s", flush=True)
            data.append(dict(engine=args.engine, tokens=n, probe=probe))
        if args.operators and not args.workload:
            ops = profile_mlx_operators(lambda: model.find(image, REQUEST), args.runs)
            print(f"operators {args.engine} N={n}: {json.dumps(ops)}", flush=True)
            data.append(dict(engine=args.engine, tokens=n, operators=ops))
        if args.profile and (args.engine != "mlx" or not args.workload):
            for mode in ("image_first", "text_first"):
                if args.engine != "mlx":
                    stages = profile_reference(model, image, mode, args.runs)
                else:
                    stages = model._mac.profile(image, REQUEST, mode, args.runs)
                print(f"profile {args.engine} N={n} {mode}: {json.dumps(stages)}", flush=True)
                data.append(dict(engine=args.engine, tokens=n, mode=mode, profile=stages))
        if args.probe_only or (args.operators_only and not args.workload):
            continue
        if args.profile_only and (args.engine != "mlx" or not args.workload):
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
            if args.operators and args.workload:
                ops = profile_mlx_operators(fn, args.runs)
                print(f"operators {args.engine} N={n} {work}: {json.dumps(ops)}", flush=True)
                data.append(dict(engine=args.engine, tokens=n, workload=work, operators=ops))
                checkpoint(data, complete=False)
                if args.operators_only:
                    del fn
                    bank = None
                    continue
            if args.profile and args.engine == "mlx" and (work not in ("ask1", "text1") or args.workload):
                stage_profile = model._mac.profile_call(fn, args.runs)
                print(f"workload profile {args.engine} N={n} {work}: {json.dumps(stage_profile)}", flush=True)
                data.append(dict(engine=args.engine, tokens=n, workload=work, profile=stage_profile))
                checkpoint(data, complete=False)
            if args.profile_only:
                del fn
                bank = None
                continue
            if args.engine == "mlx":
                import mlx.core as mx
                mx.reset_peak_memory()
            counts = dict(ask8=8, encode=1, search1=1, search256=256)
            row = measure(fn, sync, args.runs, args.warmup, trace, counts.get(work), spacing_ms=args.spacing_ms)
            if args.engine == "mlx":
                row["device_peak_bytes"] = mx.get_peak_memory()
            if bank is not None:
                row["bank_bytes"] = bank.nbytes
            floor = FLOORS[n][work]
            row.update(engine=args.engine, tokens=n, workload=work, floor_ms=floor, target_ms=floor / 0.85)
            baseline = baselines.get((n, work))
            if baseline is not None:
                row["baseline_ms"] = baseline
            data.append(row)
            checkpoint(data, complete=False)
            ms = row["median_ms"]
            warm = ", ".join(f"{t:.1f}" for t in row["warmup_ms"])
            trace_ms = f"{row['first_graph_trace_ms']:.2f}" if "first_graph_trace_ms" in row else "—"
            baseline_ms = f"{baseline:.2f}" if baseline is not None else "—"
            print(f"| {args.engine} | {n} | {work} | {baseline_ms} | {ms:.2f} | {floor / 0.85:.2f} | "
                  f"{100 * ms / floor:.1f}% | {row['first_ms']:.2f} | {trace_ms} | {warm} |", flush=True)
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
    parser.add_argument("--vision", choices=("mlx32", "mlx16", "mps32"), default="mlx32",
                        help="vision arithmetic and backend, with fp16 stored weights")
    parser.add_argument("--engine", choices=("reference", "mps16", "mlx"), default="mlx")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--spacing-ms", type=float, default=0, help="wait between timed calls outside the clock")
    parser.add_argument("--wired-mib", type=int, help="keep MLX allocations resident up to this process limit")
    parser.add_argument("--cache-mib", type=int, help="cap the free MLX allocator cache")
    parser.add_argument("--tokens", type=int, choices=FLOORS, action="append")
    parser.add_argument("--workload", choices=WORKLOADS, action="append")
    parser.add_argument("--bank-seeds", type=int, default=8)
    parser.add_argument("--reference-search-chunk", type=int, default=256)
    parser.add_argument("--trace-compile", action="store_true", help="time Python graph tracing on the first MLX invocation")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--operators", action="store_true", help="attribute MLX time to synchronized operators")
    parser.add_argument("--probe", action="store_true", help="measure native GEMM layouts and memory bandwidth")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--operators-only", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--reference-json", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.runs < 10 or args.warmup < 1:
        parser.error("use at least 10 measured runs and one warm-up run")
    if args.spacing_ms < 0:
        parser.error("--spacing-ms must be nonnegative")
    if any(n is not None and n < 0 for n in (args.wired_mib, args.cache_mib)):
        parser.error("allocator limits must be nonnegative")
    if args.engine != "mlx" and (args.wired_mib is not None or args.cache_mib is not None):
        parser.error("allocator limits require --engine mlx")
    args.profile |= args.profile_only
    args.probe |= args.probe_only
    args.operators |= args.operators_only
    if args.engine != "mlx" and (args.probe or args.operators):
        parser.error("--probe and --operators require --engine mlx")
    warnings.filterwarnings("ignore", message="CUDA is not available.*Disabling autocast")
    baselines = {}
    for path in args.reference_json:
        for row in json.loads(path.read_text())["data"]:
            if "workload" in row and "median_ms" in row:
                baselines[row["tokens"], row["workload"]] = row["median_ms"]
    if args.engine != "mlx":
        sync = torch.mps.synchronize
    else:
        import mlx.core as mx
        sync = mx.synchronize
    def load():
        with ExitStack() as stack:
            if args.engine == "mlx" and args.vision != "mlx32":
                from glancex import _mac
                from glancex._mac_layers import MPSVision, Vision
                tower = MPSVision if args.vision == "mps32" else lambda module, proc: Vision(module, proc, precise=False)
                stack.enter_context(patch.object(_mac, "Vision", tower))
            model = GlanceX.from_pretrained(args.ckpt, base=args.base, device="mps", fast=args.engine == "mlx")
        if args.engine == "mps16":
            from glancex.inference import merge_lora
            merge_lora(model.net.hf)
            model.net.half()
        if args.engine == "mlx" and model._mac is None:
            parser.error("MLX is required for --engine mlx")
        return model

    trace_ms = 0.0
    nesting = 0

    def tracked_compile(fn=None, **kwargs):
        if fn is None:
            return lambda f: tracked_compile(f, **kwargs)

        @wraps(fn)
        def tracked(*a, **kw):
            nonlocal trace_ms, nesting
            outer = nesting == 0
            start = perf_counter()
            nesting += 1
            try:
                return fn(*a, **kw)
            finally:
                nesting -= 1
                if outer:
                    trace_ms += (perf_counter() - start) * 1000

        return original_compile(tracked, **kwargs)

    with ExitStack() as stack:
        if args.cache_mib is not None:
            stack.callback(mx.set_cache_limit, mx.set_cache_limit(args.cache_mib * 2**20))
        if args.wired_mib is not None:
            wired = args.wired_mib * 2**20
            if wired > mx.device_info()["max_recommended_working_set_size"]:
                parser.error("--wired-mib exceeds the device's recommended working set")
            stack.callback(mx.set_wired_limit, mx.set_wired_limit(wired))
            stack.callback(mx.synchronize)
        trace = None
        if args.trace_compile:
            if args.engine != "mlx":
                parser.error("--trace-compile requires --engine mlx")
            original_compile = mx.compile
            stack.enter_context(patch.object(mx, "compile", tracked_compile))
            trace = lambda: trace_ms
        model, load_ms = elapsed(load, sync)
        print(f"{args.engine} model load {load_ms:.1f} ms, request block {len(model.tok.text(REQUEST))} tokens, "
              f"vision {args.vision if args.engine == 'mlx' else args.engine}", flush=True)
        print("| Engine | N | Workload | Baseline (ms) | Median (ms) | Target (ms) | % of floor | "
              "First (ms) | Trace (ms) | Warm-ups (ms) |", flush=True)
        print("|---|---:|---|---:|---:|---:|---:|---:|---:|---|", flush=True)

        def checkpoint(data, complete):
            if args.output:
                payload = dict(load_ms=load_ms, engine=args.engine, vision=args.vision, complete=complete,
                               spacing_ms=args.spacing_ms, wired_mib=args.wired_mib, cache_mib=args.cache_mib, data=data)
                args.output.write_text(json.dumps(payload, indent=2) + "\n")

        data = run(model, args, sync, checkpoint, trace, baselines)
        checkpoint(data, complete=True)


if __name__ == "__main__":
    main()
