"""Experimental Metal kernels for profiling narrow fp16 projections."""

import mlx.core as mx


_skinny = mx.fast.metal_kernel(
    name="glancex_skinny_matmul",
    input_names=["x", "weight", "shape"],
    output_names=["out"],
    source="""
    const uint lane = thread_position_in_grid.x;
    const uint col = thread_position_in_grid.y;
    const uint K4 = shape[0] / 4;
    const uint N = shape[1];
    if (col >= N) return;
    const device half4* a = reinterpret_cast<const device half4*>(x);
    const device half4* b = reinterpret_cast<const device half4*>(weight);

    float accum[M];
    #pragma unroll
    for (uint r = 0; r < M; ++r) accum[r] = 0.0f;
    for (uint k = lane; k < K4; k += 32) {
        const float4 w = float4(b[col * K4 + k]);
        #pragma unroll
        for (uint r = 0; r < M; ++r) {
            accum[r] += dot(float4(a[r * K4 + k]), w);
        }
    }

    #pragma unroll
    for (uint r = 0; r < M; ++r) {
        const float y = simd_sum(accum[r]);
        if (lane == 0) out[r * N + col] = y;
    }
    """,
)


def skinny_matmul(x, weight):
    """An fp16-weight, fp32-accumulating GEMM for short request blocks."""
    m, k = x.shape
    n = weight.shape[0]
    return _skinny(
        inputs=[x, weight, mx.array([k, n], dtype=mx.uint32)],
        template=[("M", m)], grid=(32, (n + 3) // 4 * 4, 1), threadgroup=(32, 4, 1),
        output_shapes=[(m, n)], output_dtypes=[mx.float16],
    )[0]


_skinny_mma = mx.fast.metal_kernel(
    name="glancex_skinny_mma",
    input_names=["x", "weight", "shape"],
    output_names=["partial"],
    header="#include <metal_simdgroup_matrix>\n",
    source="""
    const uint tile = thread_position_in_grid.x / 32;
    const uint col = tile * 8;
    const uint split = thread_position_in_grid.y;
    const uint K = shape[0];
    const uint N = shape[1];
    const uint M = shape[2];
    const uint first = split * CHUNK;
    const uint end = min(first + CHUNK, K);
    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = first; k < end; k += 8) {
        simdgroup_half8x8 a0, a1, b;
        simdgroup_load(a0, x + k, K);
        simdgroup_load(b, weight + col * K + k, K, ulong2(0), true);
        simdgroup_multiply_accumulate(acc0, a0, b, acc0);
        if (M == 16) {
            simdgroup_load(a1, x + 8 * K + k, K);
            simdgroup_multiply_accumulate(acc1, a1, b, acc1);
        }
    }
    simdgroup_store(acc0, partial + split * M * N + col, N);
    if (M == 16) simdgroup_store(acc1, partial + (split * M + 8) * N + col, N);
    """,
)


def skinny_mma(x, weight, chunk=256):
    m, k = x.shape
    n = weight.shape[0]
    padded = (m + 7) // 8 * 8
    a = mx.pad(x, [(0, padded - m), (0, 0)]) if padded != m else x
    splits = (k + chunk - 1) // chunk
    tiles = _skinny_mma(
        inputs=[a, weight, mx.array([k, n, padded], dtype=mx.uint32)],
        template=[("CHUNK", chunk)], grid=(n // 8 * 32, splits, 1), threadgroup=(128, 1, 1),
        output_shapes=[(splits, padded, n)], output_dtypes=[mx.float32],
    )[0]
    return tiles.sum(axis=0)[:m].astype(mx.float16)
