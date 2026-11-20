"""Metal operations for Apple-silicon inference."""

import mlx.core as mx


_delta = mx.fast.metal_kernel(
    name="glancex_delta",
    input_names=["q", "k", "v", "decay", "beta", "initial", "rows", "length"],
    output_names=["out", "final_state"],
    source="""
    // Each SIMD group owns one value column. Keeping its key axis in
    // registers avoids spilling intermediate recurrent states to memory.
    const uint lane = thread_position_in_grid.x % WIDTH;
    const uint col = thread_position_in_grid.y * (32 / WIDTH) + thread_position_in_grid.x / WIDTH;
    const uint head_row = thread_position_in_grid.z;
    if (col >= DV) return;
    const uint batch = head_row / HV;
    const uint head = head_row % HV;
    const uint key_head = head / (HV / HK);
    const uint src = (rows[batch] * HV + head) * DV * DK + col * DK + lane;
    const uint dst = head_row * DV * DK + col * DK + lane;
    const uint steps = length[0];

    float state[DK / WIDTH];
    #pragma unroll
    for (uint i = 0; i < DK / WIDTH; ++i) {
        state[i] = HAS_PREFIX ? initial[src + WIDTH * i] : 0.0f;
    }

    for (uint t = 0; t < steps; ++t) {
        const uint token = batch * steps + t;
        const uint gate = token * HV + head;
        const uint ki = (token * HK + key_head) * DK + lane;
        const uint vi = gate * DV + col;
        float keys[DK / WIDTH];
        float pred = 0.0f;
        const float factor = decay[gate];
        #pragma unroll
        for (uint i = 0; i < DK / WIDTH; ++i) {
            keys[i] = k[ki + WIDTH * i];
            state[i] *= factor;
            pred = fma(state[i], keys[i], pred);
        }
        #pragma unroll
        for (ushort step = WIDTH / 2; step > 0; step /= 2) pred += simd_shuffle_xor(pred, step);
        const float update = (float(v[vi]) - pred) * beta[gate];
        float read = 0.0f;
        #pragma unroll
        for (uint i = 0; i < DK / WIDTH; ++i) {
            state[i] = fma(keys[i], update, state[i]);
            read = fma(state[i], q[ki + WIDTH * i], read);
        }
        #pragma unroll
        for (ushort step = WIDTH / 2; step > 0; step /= 2) read += simd_shuffle_xor(read, step);
        if (lane == 0) {
            out[vi] = read;
        }
    }

    if (SAVE_STATE) {
        #pragma unroll
        for (uint i = 0; i < DK / WIDTH; ++i) {
            final_state[dst + WIDTH * i] = state[i];
        }
    }
    """,
)


def delta_rule(q, k, v, decay, beta, initial=None, rows=None, save_state=True, width=16):
    batch, length, hk, dk = q.shape
    hv, dv = v.shape[-2:]
    if rows is None:
        rows = mx.arange(batch, dtype=mx.uint32)
    state_shape = (batch, hv, dv, dk)
    return tuple(_delta(
        inputs=[q, k, v, decay, beta, initial if initial is not None else mx.zeros((1,)), rows,
                mx.array([length], dtype=mx.uint32)],
        template=[("DK", dk), ("DV", dv), ("HK", hk), ("HV", hv), ("HAS_PREFIX", initial is not None),
                  ("SAVE_STATE", save_state), ("WIDTH", width)],
        grid=(32, (dv * width + 31) // 32, batch * hv),
        threadgroup=(32, 4, 1),
        output_shapes=[v.shape, state_shape if save_state else (1,)],
        output_dtypes=[mx.float32, mx.float32],
    ))


_conv = mx.fast.metal_kernel(
    name="glancex_depthwise_silu",
    input_names=["history", "weight", "shape"],
    output_names=["out"],
    source="""
    const uint channel = thread_position_in_grid.x;
    const uint row = thread_position_in_grid.y;
    const uint channels = shape[0];
    const uint length = shape[1];
    const uint batch = row / length;
    const uint token = row % length;
    const uint offset = (batch * (length + K - 1) + token) * channels + channel;
    float sum = 0.0f;
    #pragma unroll
    for (uint tap = 0; tap < K; ++tap) {
        sum = fma(float(history[offset + tap * channels]), float(weight[channel * K + tap]), sum);
    }
    out[row * channels + channel] = sum / (1.0f + exp(-sum));
    """,
)


def depthwise_silu(history, weight):
    batch, padded_length, channels = history.shape
    kernel_size = weight.shape[1]
    length = padded_length - kernel_size + 1
    if channels % 32:
        raise ValueError("Metal depthwise convolution requires channels divisible by 32")
    return _conv(
        inputs=[history, weight, mx.array([channels, length], dtype=mx.uint32)],
        template=[("K", kernel_size)], grid=(channels, batch * length, 1), threadgroup=(32, 4, 1),
        output_shapes=[(batch, length, channels)], output_dtypes=[history.dtype],
    )[0]


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
    header="#include <metal_simdgroup_matrix>",
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
