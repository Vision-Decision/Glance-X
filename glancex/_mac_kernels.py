"""Metal operations for Apple-silicon inference."""

import mlx.core as mx


_delta = mx.fast.metal_kernel(
    name="glancex_delta",
    input_names=["q", "k", "v", "decay", "beta", "initial", "rows", "length"],
    output_names=["out", "final_state"],
    header="""
    template <int N> struct EntryType;
    template <> struct EntryType<1> { using type = float; };
    template <> struct EntryType<4> { using type = float4; };
    inline float entry_dot(float a, float b) { return a * b; }
    inline float entry_dot(float4 a, float4 b) { return dot(a, b); }
    template <class T> inline const device T* entries(const device float* p) {
        return reinterpret_cast<const device T*>(p);
    }
    template <class T> inline const constant T* entries(const constant float* p) {
        return reinterpret_cast<const constant T*>(p);
    }
    """,
    source="""
    // Each SIMD subgroup owns one value column. Keeping its key axis in
    // registers avoids spilling intermediate recurrent states to memory.
    const uint lane = thread_position_in_grid.x % WIDTH;
    const uint col = thread_position_in_grid.y * (32 / WIDTH) + thread_position_in_grid.x / WIDTH;
    const uint head_row = thread_position_in_grid.z;
    if (col >= DV) return;
    const uint batch = head_row / HV;
    const uint head = head_row % HV;
    const uint key_head = head / (HV / HK);
    constexpr uint K = DK / VEC;
    using Entry = typename EntryType<VEC>::type;
    auto qq = entries<Entry>(q);
    auto kk = entries<Entry>(k);
    auto init = entries<Entry>(initial);
    device auto* finish = reinterpret_cast<device Entry*>(final_state);
    const uint src = (rows[batch] * HV + head) * DV * K + col * K + lane;
    const uint dst = head_row * DV * K + col * K + lane;
    const uint steps = length[0];

    Entry state[K / WIDTH];
    #pragma unroll
    for (uint i = 0; i < K / WIDTH; ++i) {
        state[i] = HAS_PREFIX ? init[src + WIDTH * i] : Entry(0.0f);
    }

    for (uint t = 0; t < steps; ++t) {
        const uint token = batch * steps + t;
        const uint gate = token * HV + head;
        const uint ki = (token * HK + key_head) * K + lane;
        const uint vi = gate * DV + col;
        Entry keys[K / WIDTH];
        float pred = 0.0f;
        const float factor = decay[gate];
        #pragma unroll
        for (uint i = 0; i < K / WIDTH; ++i) {
            keys[i] = kk[ki + WIDTH * i];
            state[i] *= factor;
            pred += entry_dot(state[i], keys[i]);
        }
        #pragma unroll
        for (ushort step = WIDTH / 2; step > 0; step /= 2) pred += simd_shuffle_xor(pred, step);
        const float update = (float(v[vi]) - pred) * beta[gate];
        float read = 0.0f;
        #pragma unroll
        for (uint i = 0; i < K / WIDTH; ++i) {
            state[i] = fma(keys[i], Entry(update), state[i]);
            read += entry_dot(state[i], qq[ki + WIDTH * i]);
        }
        #pragma unroll
        for (ushort step = WIDTH / 2; step > 0; step /= 2) read += simd_shuffle_xor(read, step);
        if (lane == 0) {
            out[vi] = read;
        }
    }

    if (SAVE_STATE) {
        #pragma unroll
        for (uint i = 0; i < K / WIDTH; ++i) {
            finish[dst + WIDTH * i] = state[i];
        }
    }
    """,
)


def delta_rule(q, k, v, decay, beta, initial=None, rows=None, save_state=True, width=16, vector=1):
    batch, length, hk, dk = q.shape
    hv, dv = v.shape[-2:]
    if rows is None:
        rows = mx.arange(batch, dtype=mx.uint32)
    state_shape = (batch, hv, dv, dk)
    return tuple(_delta(
        inputs=[q, k, v, decay, beta, initial if initial is not None else mx.zeros((1,)), rows,
                mx.array([length], dtype=mx.uint32)],
        template=[("DK", dk), ("DV", dv), ("HK", hk), ("HV", hv), ("HAS_PREFIX", initial is not None),
                  ("SAVE_STATE", save_state), ("WIDTH", width), ("VEC", vector)],
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
