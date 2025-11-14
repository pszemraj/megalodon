# Internals

This document explains how the novel components of Megalodon work, focusing on implementation details.

## Moving Average Gated Attention (MEGA)

The core architectural component. Located in `megalodon/modules/moving_average_gated_attention.py:MovingAverageGatedAttention`.

### Conceptual Flow

MEGA combines three mechanisms:

1. **Complex EMA (CEMA)**: Linear-complexity sequence processing that maintains exponentially-decayed history
2. **Chunk-local attention**: Standard multi-head attention within chunk boundaries
3. **Gating**: Multiplicative interaction between EMA output and attention output

### Implementation Detail

```python
def forward(self, x, freqs_cis, mask=None, cache=None):
    # 1. TimestepNorm - running normalization
    out_tsn, prev_count, prev_mean, prev_var = self.timenorm(x, ...)

    # 2. CEMA - linear complexity sequence processing
    # Input: B x L x D -> transpose -> B x D x L
    out_cema, hx = self.cema(out_tsn.transpose(1, 2), hx)
    out_cema = out_cema.transpose(1, 2)  # B x L x D

    # 3. RMSNorm on CEMA output
    mx = self.rmsnorm(out_cema)

    # 4. Compute Q, K (with special normalization)
    z = self.wz(mx)  # B x L x zdim
    z = z.view(bsz, seq_len, num_heads, z_head_dim)
    z = self.znorm(z)  # Per-head RMSNorm
    # Split into Q and K with learned scaling
    gamma = (self.gamma + 1.0) / sqrt(z_head_dim)
    z = z.unsqueeze(2) * gamma + self.beta  # B x L x 2 x zdim
    q, k = torch.unbind(z, dim=2)

    # 5. Compute V from timestep-normed input
    v = silu(self.wv(out_tsn))  # B x L x value_dim

    # 6. Chunk-local attention
    attn = self.inner_attention(q, k, v, mask, freqs_cis, cache_attn)

    # 7. Gating with R
    r = silu(self.wr(mx))  # B x L x value_dim
    attn = attn * r

    # 8. Output projection with two-stream residual
    h = self.wh1(mx) + self.wh2(attn)
    out = h + residual  # residual = original input x
```

**Key points**:

- Q and K share a base projection (`wz`) then split with learned per-head scaling (`gamma`, `beta`)
- V comes from `out_tsn` (pre-CEMA), R comes from `mx` (post-CEMA)
- Two residual streams: CEMA output flows to Q/K/R, original input flows to V
- Attention operates on CEMA-processed queries/keys with original values

### Chunk Parallel State Passing

When `chunk_parallel_size > 1`, state must be passed between chunks. From `moving_average_gated_attention.py:247-310`:

```python
if should_recv_from_prev():
    # Receive packed tensor from previous chunk
    prev_count, prev_tensor = self._create_empty_prev_tensors(x)
    prev_tensor = recv_from_prev_chunk_parallel_region(prev_tensor)
    prev_mean, prev_var, hx = self._unpack_prev_tensors(x, prev_tensor)

# Process with received state
out_tsn, prev_count, prev_mean, prev_var = self.timenorm(x, prev_count, prev_mean, prev_var)
out_cema, hx = self.cema(out_tsn.transpose(1, 2), hx, compute_last_state=True)

if should_send_to_next():
    # Pack and send state to next chunk
    prev_tensor = self._pack_prev_tensors(prev_mean, prev_var, hx)
    prev_tensor = send_to_next_chunk_parallel_region(prev_tensor)
```

**What's passed**:
- TimestepNorm: `prev_count` (int64), `prev_mean` (G,), `prev_var` (G,) where G = num_groups
- CEMA: `hx` (D, N) complex tensor where D = model_dim, N = ndim

**Trick**: The sent tensor is added to the output with coefficient 0 to ensure backward pass is called on it:
```python
out_cema = out_cema + prev_tensor.to(out_cema).mean() * 0
```

This forces autograd to backpropagate through the chunk-parallel communication.

## Complex Exponential Moving Average (CEMA)

Located in `megalodon/modules/complex_exponential_moving_average.py:MultiHeadComplexEMA`.

### Mathematical Formulation

CEMA maintains a complex-valued hidden state and updates it via:

```
h_t = p ⊙ x_t + q ⊙ h_{t-1}
y_t = Re(γ^* h_t) + ω x_t
```

Where:
- `p`, `q`: Complex coefficients (learned)
- `h_t`: Complex hidden state (D x N)
- `γ`: Complex projection weights (D x N)
- `ω`: Real residual weight (D,)
- `⊙`: Element-wise product

### Implementation Detail

Parameters are reparameterized for stability:

```python
def coeffs(self):
    theta = sigmoid(self.theta) * (2 * pi / ndim)  # D x 1 x 1
    wavelets = arange(1, ndim + 1).view(1, ndim)
    theta = wavelets * theta  # D x N x 1

    alpha = sigmoid(self.alpha)  # D x N x 1
    delta = sigmoid(self.delta)  # D x N x 1

    p = alpha  # Real coefficient
    q = polar(1.0 - alpha * delta, theta)  # Complex: magnitude and phase
    gamma = view_as_complex(self.gamma) * scale  # D x N (complex)

    return p, q, gamma
```

**Why this form**:
- `alpha ∈ (0, 1)`: Controls input weighting
- `delta ∈ (0, 1)`: Controls decay rate
- `theta`: Phase rotation (different frequencies per dimension)
- Multi-wavelength: Each dimension gets `ndim` different frequencies

### Forward Pass

```python
def forward(self, x, hx, compute_last_state):
    # x: B x D x L
    p, q, gamma = self.coeffs()

    # Compute EMA parameters via custom kernel
    k, b = ema_parameters(p, q, gamma, hx, seq_len)
    # k: convolution kernel (L,)
    # b: bias from initial state (B, D) if hx provided

    # Apply via FFT convolution
    output = fftconv(x, k)  # B x D x L
    if b is not None:
        output = output + b

    # Compute final hidden state if needed
    if compute_last_state:
        h = ema_hidden(x, p, q, hx)  # B x D x N (complex)

    # Residual
    output = output + x * self.omega

    return output, h
```

**Custom kernels**:
1. `ema_parameters`: Computes convolution kernel from EMA coefficients
   - Uses recurrence relation to build kernel: `k[t] = p * q^t`
   - Applies gamma projection
   - Handles initial state `hx` to produce bias `b`

2. `fftconv`: FFT-based convolution
   - Pads sequence to power of 2
   - FFT → multiply → iFFT
   - Faster than direct convolution for long sequences

3. `ema_hidden`: Computes final hidden state
   - Iterates through sequence: `h[t] = p * x[t] + q * h[t-1]`
   - Only called when state needs to be passed to next chunk

**Performance hack**: `ema_parameters` and `fftconv` are fused when possible to avoid materializing large kernels.

### Initialization

From `complex_exponential_moving_average.py:18-40`:

```python
def _reset_parameters(alpha, delta, theta, gamma, omega, embed_dim):
    # alpha, delta: Small random values
    init.normal_(alpha, mean=0.0, std=0.2)
    init.normal_(delta, mean=0.0, std=0.2)

    # theta: Log-spaced frequencies (global permutation for diversity)
    freqs = exp(arange(1, embed_dim + 1) * -log(embed_dim) / embed_dim)
    freqs = freqs[random_permutation(embed_dim)]  # shuffled
    theta = log(freqs / (1.0 - freqs))  # logit space

    # gamma: Random with imaginary part = 0
    init.normal_(gamma[:, :, 0], std=1.0)  # real part
    gamma[:, :, 1] = 0.0  # imaginary part

    # omega: Small random
    init.trunc_normal_(omega, std=0.25, a=-1.0, b=1.0)
```

**Why**:
- Frequencies are permuted to decorrelate dimensions
- Imaginary part of gamma starts at 0 for stability
- Small initialization prevents exploding hidden states

## TimestepNorm

Located in `megalodon/modules/timestep_norm.py:TimestepNorm`.

Standard normalization computes statistics over spatial dimensions (batch, features). TimestepNorm computes **running statistics across time** within a chunk, similar to BatchNorm but along the sequence dimension.

### Formulation

For sequence `x_1, x_2, ..., x_L` (L timesteps):

```
μ_t = (count_{t-1} * μ_{t-1} + x_t) / count_t
σ²_t = (count_{t-1} * σ²_{t-1} + (x_t - μ_t)²) / count_t
count_t = count_{t-1} + 1

y_t = γ * (x_t - μ_t) / sqrt(σ²_t + ε) + β
```

Where:
- `μ_t`, `σ²_t`: Running mean and variance up to timestep t
- `count_t`: Number of timesteps seen
- `γ`, `β`: Learned affine parameters

**Group variant**: When `num_groups < model_dim`, features are split into groups and each group has independent statistics (like GroupNorm).

### Implementation

Forward pass (simplified from `timestep_norm.py:106`):

```python
def forward(self, x, prev_count=None, prev_mean=None, prev_var=None, padding_mask=None):
    # x: B x L x D
    if prev_count is None:
        prev_count = self.prior_count  # Usually 0
        prev_mean = self.prior_mean    # Learned or 0
        prev_var = self.prior_logv.exp()  # Learned or 0

    # Custom CUDA kernel
    y, count, mean, var, cummean, cumrstd = group_timestep_norm_fwd(
        x, prev_count, prev_mean, prev_var,
        self.weight + 1.0,  # gamma
        self.bias,          # beta
        padding_mask,
        self.groups_per_partition,
        self.eps
    )

    return y, count, mean, var
```

**CUDA kernel** (`megalodon/csrc/ops/timestep_norm_kernel.cu`):
- Iterates through sequence, updating running stats
- Uses Welford's algorithm for numerical stability
- Saves cumulative stats (`cummean`, `cumrstd`) for backward pass
- Supports masking to skip padding tokens

**Backward pass**:
- Backprop through normalization using saved cumulative stats
- Gradients for `prev_mean`, `prev_var` propagate to previous chunk

### Use in Chunk Parallel

Each chunk receives `(prev_count, prev_mean, prev_var)` from the previous chunk:

```python
# Chunk 0: count=0, mean=0, var=init
# Chunk 1: count=2048, mean=..., var=... (from chunk 0)
# Chunk 2: count=4096, mean=..., var=... (from chunk 1)
```

This allows the model to maintain global statistics across an entire long sequence.

**Prior count**: For non-chunk-parallel training, `prior_count > 0` acts as a regularizer (Bayesian prior on statistics).

## Efficient Attention

Located in `megalodon/modules/fused_ops/attention/swift.py`.

### Swift Attention

Flash-attention-style fused kernel. Key idea: compute attention in blocks without materializing full attention matrix.

```python
def swift_efficient_attention(q, k, v, scale, dropout, causal, training):
    # q, k, v: B x L x H x D
    # Forward: calls swift_efficient_attention_fwd (custom CUDA)
    # Backward: calls swift_efficient_attention_bwd (custom CUDA)
```

**Forward kernel** (`megalodon/csrc/ops/attention_kernel.cu`):
- Tiles Q, K, V into blocks (typically 64x64)
- For each Q tile:
  - Loads K, V tiles
  - Computes `QK^T` in registers
  - Applies softmax (numerically stable with max subtraction)
  - Multiplies by V
  - Accumulates to output
- Never materializes full `B x H x L x L` attention matrix

**Causal masking**:
- When `causal=True`, only computes lower-triangular part of attention
- Skip K tiles where `k_col > q_row`

**Backward kernel**:
- Recompute forward pass blocks on-the-fly (activation recomputation)
- Compute gradients w.r.t. Q, K, V using chain rule
- Fused operations to minimize memory traffic

**Advantage**: Memory usage O(L) instead of O(L²), enabling longer sequences.

## FFT Convolution

Located in `megalodon/modules/fused_ops/fftconv.py`.

Used by CEMA to apply convolution kernels efficiently.

### Algorithm

For input `x` (B, D, L) and kernel `k` (L,):

```
y = x ⊛ k  # convolution
```

Naive: O(L²) time. FFT-based: O(L log L) time.

```python
def fftconv(x, k):
    # 1. Pad to power of 2 for efficient FFT
    N = next_power_of_2(L)
    x_pad = pad(x, (0, N - L))
    k_pad = pad(k, (0, N - L))

    # 2. FFT both inputs
    X = fft(x_pad)  # B x D x N (complex)
    K = fft(k_pad)  # N (complex)

    # 3. Pointwise multiply
    Y = X * K.unsqueeze(0).unsqueeze(0)

    # 4. Inverse FFT
    y = ifft(Y).real  # B x D x N

    # 5. Truncate to original length
    y = y[:, :, :L]

    return y
```

**Custom kernel** (`megalodon/csrc/ops/fftconv_kernel.cu`):
- Uses cuFFT library for FFT/iFFT
- Fuses padding, FFT, multiply, iFFT into single kernel launch
- Supports batched FFTs for efficiency
- Real-to-complex FFT optimization (input is real)

**Fused variant** (`fused_fftconv_fwd`):
- Combines `ema_parameters` and `fftconv` into single operation
- Avoids materializing kernel `k` explicitly
- Further memory and compute savings

## Model Parallelism

Tensor parallelism implementation in `megalodon/modules/model_parallel/layers.py`.

### Column Parallel Linear

Splits output dimension across GPUs:

```python
class ColumnParallelLinear:
    def __init__(self, input_dim, output_dim, gather_output, ...):
        world_size = get_model_parallel_world_size()
        self.output_dim_per_partition = output_dim // world_size
        self.weight = Parameter(torch.empty(self.output_dim_per_partition, input_dim))

    def forward(self, x):
        # x: replicated across GPUs (if input_is_parallel=False)
        if not self.input_is_parallel:
            # All ranks have same x
            output_parallel = F.linear(x, self.weight, self.bias)
        else:
            # x is already partitioned
            output_parallel = F.linear(x, self.weight, self.bias)

        if self.gather_output:
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel

        return output
```

**Key functions** (`megalodon/modules/model_parallel/mappings.py`):
```python
def gather_from_model_parallel_region(tensor):
    # All-gather along last dimension
    world_size = get_model_parallel_world_size()
    gathered = torch.empty(..., tensor.size(-1) * world_size, ...)
    all_gather(gathered, tensor, group=get_model_parallel_group())
    return gathered

def scatter_to_model_parallel_region(tensor):
    # Scatter along last dimension
    rank = get_model_parallel_rank()
    world_size = get_model_parallel_world_size()
    dim_per_rank = tensor.size(-1) // world_size
    return tensor[..., rank * dim_per_rank:(rank + 1) * dim_per_rank]
```

### Row Parallel Linear

Splits input dimension across GPUs:

```python
class RowParallelLinear:
    def __init__(self, input_dim, output_dim, ...):
        world_size = get_model_parallel_world_size()
        self.input_dim_per_partition = input_dim // world_size
        self.weight = Parameter(torch.empty(output_dim, self.input_dim_per_partition))

    def forward(self, x):
        # x is partitioned across last dimension
        output_parallel = F.linear(x, self.weight, None)  # No bias yet

        # All-reduce across model parallel group
        output = reduce_from_model_parallel_region(output_parallel)

        # Add bias on replicated output
        if self.bias is not None:
            output = output + self.bias

        return output
```

**reduce_from_model_parallel_region**:
```python
def reduce_from_model_parallel_region(tensor):
    all_reduce(tensor, op=ReduceOp.SUM, group=get_model_parallel_group())
    return tensor
```

### Example: MLP with Model Parallelism

```
Input: [B, L, D] (replicated across all ranks)
  ↓
ColumnParallelLinear (D → H):
  Rank 0: weight [H/2, D] → output [B, L, H/2]
  Rank 1: weight [H/2, D] → output [B, L, H/2]
  (No communication, outputs are partitioned)
  ↓
Activation (SiLU): Element-wise, no communication
  ↓
RowParallelLinear (H → D):
  Rank 0: weight [D, H/2] → partial [B, L, D]
  Rank 1: weight [D, H/2] → partial [B, L, D]
  All-reduce(partial) → output [B, L, D] (replicated)
```

**Communication cost**: One all-reduce per MLP block (RowParallel layer).

## Custom Dropout

`memory_efficient_dropout` in `megalodon/modules/fused_ops/memory_efficient_dropout.py`:

```python
def memory_efficient_dropout(x, p, training):
    if not training or p == 0:
        return x
    # Custom backward that regenerates mask instead of saving it
    return MemoryEfficientDropoutFunc.apply(x, p)
```

**Standard dropout**: Saves binary mask for backward pass → O(n) memory.

**Memory-efficient dropout**: Regenerates mask using same random seed in backward → O(1) memory.

Implementation:
```python
class MemoryEfficientDropoutFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p):
        mask = (torch.rand_like(x) > p) / (1 - p)
        ctx.save_for_backward(torch.tensor(p))
        return x * mask

    @staticmethod
    def backward(ctx, grad_output):
        p = ctx.saved_tensors[0].item()
        # Regenerate mask with same random state (implementation detail)
        mask = (torch.rand_like(grad_output) > p) / (1 - p)
        return grad_output * mask, None
```

**Caveat**: Requires careful random state management to ensure same mask is regenerated. The actual implementation uses custom CUDA kernels with controlled PRNG state.

## Performance Optimizations

1. **Fused kernels**: TimestepNorm, attention, EMA operations fused to reduce kernel launches
2. **Mixed precision**: BF16 activations with FP32 accumulation in critical paths
3. **Activation recomputation**: Flash attention recomputes forward pass in backward to save memory
4. **FFT convolution**: O(L log L) instead of O(L²) for CEMA
5. **Memory-efficient dropout**: Saves O(n) memory by regenerating masks
6. **Tensor parallelism**: Overlaps compute and communication via async all-reduce
7. **FSDP**: Shards optimizer states and gradients across data parallel ranks

## Numerical Stability

Several tricks ensure stable training:

1. **Welford's algorithm** in TimestepNorm: Numerically stable mean/variance updates
2. **Sigmoid reparameterization** in CEMA: Forces `alpha`, `delta` ∈ (0, 1)
3. **RMSNorm on Q/K**: Normalizes per-head before attention to prevent extreme scales
4. **Learned scaling** (`gamma`, `beta`): Allows model to control Q/K magnitude
5. **FP32 reduce-scatter** option in FSDP: Accumulates gradients in FP32 despite BF16 training

---

## Complete CUDA Kernel Reference

This section documents all custom CUDA kernels in detail.

### Kernel Catalog

The `megalodon_extension` module exposes 7 custom operations (from `megalodon/csrc/megalodon_extension.cc`):

| Operation | Python Wrapper | CUDA Implementation | Status |
|-----------|---------------|---------------------|---------|
| **TimestepNorm** | `megalodon/modules/timestep_norm.py` | `csrc/ops/timestep_norm*` | Used |
| **EMAHidden** | `megalodon/modules/fused_ops/ema_hidden.py` | `csrc/ops/ema_hidden*` | Used |
| **EMAParameters** | `megalodon/modules/fused_ops/ema_parameters.py` | `csrc/ops/ema_parameters*` | Used |
| **FFTConv** | `megalodon/modules/fused_ops/fftconv.py` | `csrc/ops/fftconv*` | Used |
| **Attention** | `megalodon/modules/fused_ops/attention/swift.py` | `csrc/ops/attention*` | Used |
| **AttentionSoftmax** | None | `csrc/ops/attention_softmax*` | Unused |
| **SequenceNorm** | None | `csrc/ops/sequence_norm*` | Unused |

### 1. TimestepNorm Kernel

**Purpose**: Compute running mean and variance across sequence timesteps with group normalization.

**C++ Signature** (`csrc/ops/timestep_norm.h`):
```cpp
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
GroupTimestepNormFwd(
    const torch::Tensor& x,               // [B, L, D]
    const torch::Tensor& prev_count,      // [B]
    const torch::Tensor& prev_mean,       // [B, G]
    const torch::Tensor& prev_var,        // [B, G]
    const torch::Tensor& gamma,           // [D]
    const torch::Tensor& beta,            // [D]
    const torch::Tensor& padding_mask,    // [B, L] or empty
    int64_t num_groups,
    double eps
);
```

**Algorithm** (`csrc/ops/timestep_norm_kernel.cu`):

1. **Welford's online algorithm** for mean/variance:
   ```
   For each timestep t:
     count_t = count_{t-1} + 1
     delta = x_t - mean_{t-1}
     mean_t = mean_{t-1} + delta / count_t
     M2_t = M2_{t-1} + delta * (x_t - mean_t)
     var_t = M2_t / count_t
   ```

2. **Group-wise statistics**: Features split into `num_groups`, each with independent stats

3. **Normalization**:
   ```
   y_t = gamma * (x_t - mean_t) / sqrt(var_t + eps) + beta
   ```

4. **Cumulative stats**: Stores `cummean[t]`, `cumrstd[t]` for backward pass

**Numerical stability**:
- Welford's algorithm avoids catastrophic cancellation
- Uses Kahan summation for accumulating across groups
- Computes reciprocal std once and reuses: `rstd = 1 / sqrt(var + eps)`

**Memory saved for backward**:
- `cummean`: Cumulative mean at each timestep [B, L, G]
- `cumrstd`: Cumulative reciprocal std at each timestep [B, L, G]
- Does not save normalized output or intermediate deltas

### 2. EMA Hidden State Kernel

**Purpose**: Compute final hidden state of complex EMA after processing sequence.

**C++ Signature** (`csrc/ops/ema_hidden.h`):
```cpp
std::tuple<torch::Tensor, torch::Tensor> EMAHiddenFwd(
    const torch::Tensor& x,        // [B, D, L] - input sequence
    const torch::Tensor& p,        // [D, N, 1] - real input coefficient
    const torch::Tensor& log_q,    // [D, N, 1] - log of decay coefficient
    const torch::Tensor& hx        // [B, D, N] - initial hidden state (complex)
);
// Returns: (h, v) where h = final hidden state, v = Vandermonde matrix
```

**Algorithm**:

1. **Recurrence relation**:
   ```
   h[0] = hx (if provided, else 0)
   For t = 1 to L:
     h[t] = p * x[t] + q * h[t-1]
   ```
   Where `p`, `q` are complex numbers, `x` is real.

2. **Chunked processing**:
   - Processes sequence in blocks of 64-256 timesteps
   - Uses Vandermonde matrix to batch-compute powers of q: `[q^0, q^1, ..., q^n]`
   - Applies prefix sum with complex multiplication

3. **Complex arithmetic**:
   - Uses `c10::complex<float>` for complex numbers
   - Fused multiply-add: `h = p * x + q * h_prev`
   - Stores in view-as-real format: `[..., 2]` where last dim is `[real, imag]`

**Backward pass**:
- Reverse-mode recurrence: `grad_h[t-1] = q * grad_h[t]`
- Gradients for p, q computed via chain rule
- Vandermonde matrix reused from forward pass

**Optimizations**:
- Uses shared memory for Vandermonde matrix
- Warp-level primitives for reduction
- Processes multiple batches/dimensions in parallel

### 3. EMA Parameters Kernel

**Purpose**: Compute convolution kernel and bias from EMA coefficients.

**C++ Signature** (`csrc/ops/ema_parameters.h`):
```cpp
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> EMAParametersFwd(
    const torch::Tensor& p,        // [D, N, 1]
    const torch::Tensor& log_q,    // [D, N, 1]
    const torch::Tensor& gamma,    // [D, N] (complex)
    const torch::Tensor& hx,       // [B, D, N] (complex) or empty
    int64_t length
);
// Returns: (weight, bias, vander)
// weight: [D, L] - convolution kernel
// bias: [B, D] or empty - contribution from initial state
// vander: [D, N, L] - Vandermonde matrix for backward
```

**Algorithm**:

1. **Kernel computation**:
   ```
   k[t] = sum_n gamma[n] * p[n] * q[n]^t   for t = 0 to L-1
   ```
   This is the impulse response of the EMA filter.

2. **Vandermonde matrix**:
   ```
   vander[n, t] = q[n]^t
   ```
   Computed efficiently using recurrence: `vander[t] = vander[t-1] * q`

3. **Matrix multiplication**:
   ```
   k = gamma @ vander  # [D, N] @ [D, N, L] -> [D, L]
   k = k * p           # Element-wise multiply
   ```

4. **Bias from initial state**:
   ```
   If hx is provided:
     bias[b, d] = sum_n gamma[n] * hx[b, d, n]
   ```

**Why this works**:
- EMA recurrence can be unrolled into a convolution
- The kernel is exponentially decaying (since |q| < 1)
- FFT convolution in next step applies this kernel efficiently

**Numerical stability**:
- Works in log-space for q (`log_q`) to avoid underflow
- Converts to linear space only when needed: `q = exp(log_q)`
- Kahan summation for reducing across `n` dimension

### 4. FFTConv Kernel

**Purpose**: Apply convolution via FFT for O(L log L) complexity.

**C++ Signature** (`csrc/ops/fftconv.h`):
```cpp
std::tuple<torch::Tensor, torch::Tensor> FFTConvFwd(
    const torch::Tensor& x,        // [B, D, L]
    const torch::Tensor& k_f       // [D, N_fft] (complex) - already FFT'd kernel
);
// Returns: (y, x_f) where y = output, x_f = FFT of x (for backward)
```

**Algorithm**:

1. **Padding**: Pad sequence to 2*N where N is next power of 2 ≥ L
   - This prevents circular convolution artifacts

2. **Forward FFT**: `X = FFT(x)` using cuFFT
   - Real-to-complex FFT (saves memory)
   - Batched across B and D dimensions

3. **Pointwise multiply**: `Y = X * K`
   - Element-wise complex multiplication
   - K is already in frequency domain

4. **Inverse FFT**: `y = IFFT(Y)`
   - Complex-to-real iFFT
   - Truncate to original length L

**Optimizations** (`csrc/fft.cuh`):
- **Custom FFT for small sizes**: For L ≤ 256, uses hand-coded FFT kernel (faster than cuFFT)
- **Twiddle factor lookup**: Precomputed twiddle factors in LUT (`twiddle_factor_lut.cuh`)
- **Register-based computation**: Small FFTs computed entirely in registers
- **Shared memory**: Larger FFTs use shared memory for transpose

**Backward pass**:
```
grad_x = IFFT(FFT(flip(grad_y)) * K)
grad_k = IFFT(FFT(flip(grad_y)) * X)
```
Where `flip` reverses the sequence (correlation = flipped convolution).

**Dispatch logic** (`megalodon/modules/fused_ops/fftconv.py:79-93`):
- If `not x.is_cuda` or `L < 32` or `L > 16384`: Use PyTorch FFT
- Else: Use custom CUDA kernel

This explains the 16384 max length mentioned in docs.

### 5. Attention Kernel (Swift)

**Purpose**: Flash-attention-style fused attention with memory efficiency.

**C++ Signature** (`csrc/ops/attention.h`):
```cpp
std::tuple<torch::Tensor, torch::Tensor> AttentionCUDAFwd(
    const torch::Tensor& q,           // [B, L, H, D]
    const torch::Tensor& k,           // [B, L, H, D]
    const torch::Tensor& v,           // [B, L, H, Dv]
    double scale,                     // 1.0 typically
    double dropout,
    bool use_causal_mask
);
// Returns: (output, attention_weights)
// attention_weights only computed if dropout > 0 (for backward)
```

**Algorithm** (`csrc/ops/attention_kernel.cu`):

1. **Tiling**:
   - Q split into blocks of size `[Bq, Dq]` (typically 64x64)
   - K, V split into blocks of size `[Bk, Dk]`
   - Processes Q tiles in outer loop, K tiles in inner loop

2. **Per-tile computation**:
   ```
   For each Q tile:
     Load Q tile into shared memory
     For each K tile:
       Load K, V tiles into shared memory
       Compute S = Q @ K^T in registers
       Apply scaling and optional causal mask
       Compute softmax(S) using online algorithm
       Accumulate: out += softmax(S) @ V
   ```

3. **Online softmax**:
   ```
   For each row of Q:
     max = -inf, sum = 0, acc = 0
     For each K tile:
       scores = Q[row] @ K^T
       new_max = max(max, max(scores))
       # Rescale previous accumulator
       acc *= exp(max - new_max)
       sum *= exp(max - new_max)
       # Add new contribution
       acc += exp(scores - new_max) @ V
       sum += sum(exp(scores - new_max))
       max = new_max
     # Final normalization
     out = acc / sum
   ```

4. **Causal masking**:
   - When `use_causal_mask=True`, skip K tiles where `col_idx > row_idx`
   - Apply mask within tile by setting scores to `-inf` before softmax

**Memory complexity**:
- Does not materialize full `[B, H, L, L]` attention matrix
- Only stores one tile at a time in shared memory
- Total memory: O(L) instead of O(L²)

**Backward pass**:
- Recomputes forward pass on-the-fly (activation recomputation)
- Computes gradients tile-by-tile
- Uses same tiling strategy as forward

**Differences from FlashAttention-2**:
- Uses simpler tiling (no work partitioning across warps)
- No sequence-parallel reduction
- Fewer tuning parameters
- Adequate for Megalodon's chunk-local attention (max 2048 tokens)

### 6. Attention Softmax Kernel (Unused)

**Purpose**: Fused attention with explicit softmax computation.

**Status**: Python wrapper exists (`megalodon/modules/fused_ops/attention/softmax.py`) but is never imported or used. The model uses `swift_efficient_attention` instead.

**Implementation**: Similar to regular attention kernel but always materializes softmax output for debugging.

### 7. Sequence Norm Kernel (Unused)

**Purpose**: Alternative to TimestepNorm, possibly layer normalization across sequence.

**Status**: CUDA implementation exists (`csrc/ops/sequence_norm*`) but no Python wrapper. Not registered in the module's `__init__.py`.

**Speculation**: May have been an earlier normalization scheme that was replaced by TimestepNorm.

---

## Utility Headers and Infrastructure

The kernels rely on several utility headers that provide common functionality.

### Welford's Algorithm (`csrc/welford.h`)

**Purpose**: Numerically stable online computation of mean and variance.

**Data structure**:
```cpp
template <typename T>
struct WelfordData {
  int64_t m0 = 0;  // Count
  T m1 = T(0);     // Mean
  T m2 = T(0);     // Sum of squared deviations
};
```

**Update rule**:
```cpp
data += x:
  m0 += 1
  delta1 = x - m1
  m1 += delta1 / m0
  delta2 = delta1 * (x - m1) - m2
  m2 += delta2 / m0
```

**Variance**: `var = m2 / m0`

**Why stable**: Avoids `E[x²] - E[x]²` formula which suffers from catastrophic cancellation.

**Used in**: TimestepNorm, SequenceNorm

### Kahan Summation (`csrc/kahan.h`)

**Purpose**: Compensated summation to reduce numerical error in accumulation.

**Algorithm**:
```cpp
KahanAdd(x, sum, compensation):
  y = x - compensation
  t = sum + y
  compensation = (t - sum) - y
  return (t, compensation)
```

**Error bound**: O(ε) instead of O(nε) for n additions.

**Used in**: TimestepNorm group reduction, EMA parameter accumulation

**Wrapper class**: `KahanWrapper<T>` provides `+=` operator with automatic compensation tracking.

### Complex Number Utilities (`csrc/complex_utils.cuh`)

**Purpose**: CUDA device functions for complex arithmetic.

**Key functions**:
- `Mul1i(z)`: Multiply by i: `-imag + i*real`
- `RealOfProduct(z1, z2)`: Real part of `z1 * z2` without computing imaginary
- `Exp(z)`: Complex exponential using `e^real * (cos(imag) + i*sin(imag))`

**Uses**: `sincos` intrinsic for efficient simultaneous sin/cos computation.

**Used in**: CEMA operations, EMA kernels

### FFT Utilities (`csrc/fft.cuh`)

**Purpose**: Custom FFT implementations for small sizes.

**Features**:
- `Log2(x)`: Compile-time log2 via switch statement
- `LoadAsComplex`: Load real data as complex with zero imaginary part
- Radix-2 FFT kernel: Cooley-Tukey algorithm for powers of 2
- Twiddle factor computation on-the-fly or from LUT

**Constants**:
- `kFFTMaxLength = 16384`: Maximum sequence length for custom FFT
- `kFFTNumThreads = 1024`: Threads per block for FFT

**Twiddle factors** (`csrc/twiddle_factor.cuh`, `csrc/twiddle_factor_lut.cuh`):
```cpp
W_N^k = exp(-2πi * k / N)
```
Precomputed for common sizes, computed on-the-fly for others.

**Used in**: FFTConv kernel for small sequences

### CUDA Utilities (`csrc/cuda_utils.cuh`)

**Purpose**: Common CUDA device functions and macros.

**Likely contents** (not read in full):
- Warp shuffle primitives
- Reduction patterns (sum, max, etc.)
- Block-level synchronization
- Shared memory indexing helpers

### Reduction Utilities (`csrc/reduce.cuh`)

**Purpose**: Efficient parallel reduction primitives.

**Patterns**:
- Warp-level reduce (shuffle-based)
- Block-level reduce (shared memory + tree reduction)
- Grid-level reduce (atomic operations or multi-kernel)

**Used in**: All kernels for computing sums, max, mean across dimensions

### Register Utilities (`csrc/register_utils.cuh`)

**Purpose**: Helpers for managing register-level data.

**Likely functions**:
- Array unrolling
- Register spilling prevention
- Template metaprogramming for compile-time loop unrolling

### Softmax Utilities (`csrc/softmax.cuh`)

**Purpose**: Numerically stable softmax implementations.

**Algorithm**:
```
max = max(x)
exp_sum = sum(exp(x - max))
softmax(x) = exp(x - max) / exp_sum
```

**Variants**:
- Row-wise softmax (for attention)
- Masked softmax (for causal masking)
- Online softmax (for flash attention)

**Used in**: Attention kernels

### Random Utilities (`csrc/random_utils.cuh`)

**Purpose**: Fast random number generation on GPU.

**Uses**: Likely cuRAND or custom PRNG for dropout masks.

**Requirements**:
- Reproducible: Same seed → same mask
- Fast: Must not bottleneck training
- Per-thread state: Each thread has independent PRNG

**Used in**: Memory-efficient dropout (to regenerate masks in backward)

### BLAS Wrappers (`csrc/blas.cc`, `csrc/blas.h`)

**Purpose**: C++ wrappers for cuBLAS operations.

**Likely operations**:
- Matrix multiplication (GEMM)
- Batched matrix multiplication
- Strided batched operations

**Used in**: Possibly for large matrix operations not handled by PyTorch

---

## Compilation and Build System

### Extension Registration (`csrc/megalodon_extension.cc`)

Registers all operations with PyBind11:

```cpp
PYBIND11_MODULE(megalodon_extension, m) {
  py::module m_ops = m.def_submodule("ops", "Submodule for custom ops.");
  ops::DefineAttentionOp(m_ops);
  ops::DefineAttentionSoftmaxOp(m_ops);
  ops::DefineEMAHiddenOp(m_ops);
  ops::DefineEMAParametersOp(m_ops);
  ops::DefineFFTConvOp(m_ops);
  ops::DefineSequenceNormOp(m_ops);
  ops::DefineTimestepNormOp(m_ops);
}
```

Each `Define*Op` function binds the C++ functions to Python-callable names.

### Build Configuration (`setup.py`)

**Compilation flags**:

C++:
- `-O3`: Maximum optimization
- `-std=c++17`: C++17 standard (for structured bindings, etc.)

NVCC:
- `--expt-relaxed-constexpr`: Allow constexpr in device code
- `--expt-extended-lambda`: Allow extended lambda syntax
- `--threads 4`: Parallel compilation

**Include paths**: `megalodon/csrc` for headers

**Source files**: All `.cc` and `.cu` files in `csrc/` and `csrc/ops/`

### Dispatch Pattern

Most kernels follow this pattern:

1. **Python wrapper** (`megalodon/modules/fused_ops/*.py`):
   - Defines `torch.autograd.Function` with `forward()` and `backward()`
   - Calls into C++ extension

2. **C++ dispatcher** (`csrc/ops/*.cc`):
   - Validates inputs (dtypes, shapes, device)
   - Dispatches to CPU or CUDA implementation
   - Handles dtype conversion if needed

3. **CUDA kernel** (`csrc/ops/*_kernel.cu`):
   - Template kernels for different dtypes
   - Launch configuration (grid, block dimensions)
   - Actual computation

Example from TimestepNorm:
```
Python: timestep_norm.py:TimestepNormFunc.forward()
  ↓
C++: ops/timestep_norm.cc:GroupTimestepNormFwd()
  ↓ (if x.is_cuda())
CUDA: ops/timestep_norm_kernel.cu:GroupTimestepNormCUDAFwdKernel<<<>>>()
```

---

## Why Custom Kernels?

Standard PyTorch operations don't cover Megalodon's needs:

1. **TimestepNorm**: No PyTorch equivalent for running normalization across time
2. **CEMA**: Complex-valued recurrence with specific structure
3. **EMA Parameters**: Vandermonde matrix construction with complex coefficients
4. **FFTConv**: Fused FFT + convolution with custom handling
5. **Attention**: Memory-efficient flash attention variant

**Performance gains**:
- Kernel fusion: Multiple ops in one kernel launch (reduced memory traffic)
- Custom memory layout: Optimized for access patterns
- Reduced materialization: Don't store intermediate results
- Better numerical stability: Specialized algorithms (Welford, Kahan, online softmax)

**Estimated speedup** (vs. PyTorch native):
- TimestepNorm: 3-5x (vs. manual loops in PyTorch)
- FFTConv: 2x (vs. PyTorch FFT + multiply + iFFT)
- Attention: 3-10x (vs. materialized attention matrix)
- CEMA: 5-10x (vs. Python loops with complex numbers)

These are rough estimates; actual speedup depends on sequence length, batch size, and hardware.
