# Architecture

This document explains how the Megalodon codebase is structured, following the execution path from entry points through the core model components.

## Entry Points

### Evaluation: `eval.py`

The primary executable entry point. Flow:

1. **Initialization** (`eval.py:main`):
   ```python
   # Parse command-line args via fire
   # Initialize distributed training (torch.distributed)
   # Setup model parallelism via initialize_model_parallel()
   # Reload model checkpoint via reload_model()
   ```

2. **Model Loading** (`megalodon/reloading.py:reload_model`):
   - Loads model config from checkpoint directory
   - Builds model via `build_model()` with FSDP wrapping
   - Loads tokenizer (SentencePiece)
   - Loads model state dict from checkpoint

3. **Evaluation Tasks**:
   - **Perplexity** (`eval.py:eval_ppl`): Loads JSONL data, runs inference, computes cross-entropy loss
   - **Generation** (`eval.py:run_generation`): Loads prompts, runs autoregressive generation, saves outputs

### Training: Pseudo-code Only

The README shows pseudo-code for training (lines 82-152), but no actual training script exists. The pattern is:

```python
model = build_model(cfg.model, dtype, ...)
optimizer, scheduler = build_optimizer(model, cfg.optim, ...)  # not implemented

for batch in dataloader:
    pred, _ = model(x)
    loss = cross_entropy(pred, y).mean()
    loss.backward()
    model.grad_all_reduce()  # sync chunk parallel grads
    clip_grad_norm_(model, max_norm=cfg.optim.clip)
    optimizer.step()
```

Key point: `build_optimizer` and the training loop are **not implemented** in this repo.

## Module Structure

```
megalodon/
├── config.py              # Dataclass configs (ModelConf, OptimConf, ValidConf)
├── model/
│   └── mega.py           # Core model: Mega, MegaBlock, build_model()
├── modules/
│   ├── moving_average_gated_attention.py    # MEGA attention layer
│   ├── complex_exponential_moving_average.py # CEMA (linear complexity component)
│   ├── normalized_feedforward_network.py    # FFN with pre-norm
│   ├── timestep_norm.py                     # Running normalization across time
│   ├── rotary_positional_embedding.py       # RoPE for attention
│   ├── layer_norm.py                        # Fused LayerNorm/RMSNorm
│   ├── model_parallel/                      # Tensor parallel layers
│   ├── chunk_parallel/                      # Chunk parallel communication
│   └── fused_ops/                           # Custom CUDA ops
├── distributed/
│   ├── initialize.py      # Setup model/data/chunk parallel groups
│   ├── fairscale_fsdp.py  # FSDP wrapper
│   └── setup.py           # torch.distributed initialization
├── data/
│   ├── tokenizer.py       # SentencePiece wrapper
│   └── dataloader.py      # JSONL data loader
├── generation.py          # Autoregressive generation
├── inference.py           # Batch inference for perplexity
└── utils.py               # Helpers: dtype conversion, caching, etc.
```

## Data Flow

### Forward Pass (Training/Evaluation)

Starting from `megalodon/model/mega.py:Mega.forward()`:

```
Input: tokens [batch, seq_len]
  ↓
1. Chunk splitting (if chunk_parallel_size > 1)
   - Each rank gets seq_len / chunk_parallel_size tokens
  ↓
2. Embedding
   - ParallelEmbedding (model parallel)
   - Optional scaling by sqrt(model_dim)
   - Dropout
  ↓
3. RoPE frequency computation
   - Per-chunk frequencies for rotary embeddings
  ↓
4. Layer-wise processing (for each MegaBlock)
   ├─> MovingAverageGatedAttention
   │   ├─> TimestepNorm (running stats from previous chunk if chunk_parallel)
   │   ├─> CEMA (linear complexity sequence processing)
   │   ├─> RMSNorm
   │   ├─> Q/K/V projections (model parallel)
   │   ├─> _InnerAttention (chunk-local attention with RoPE)
   │   └─> Output projection (model parallel)
   └─> NormalizedFeedForwardNetwork
       ├─> LayerNorm
       ├─> FC1 (model parallel)
       ├─> SiLU activation (+ optional SwiGLU gate)
       └─> FC2 (model parallel)
  ↓
5. Output layer
   - TimestepNorm (with chunk parallel state passing)
   - RowParallelLinear to vocab_size
  ↓
Output: logits [batch, seq_len, vocab_size]
```

### Chunk Parallel Communication

When `chunk_parallel_size > 1`, state must be passed between chunks:

```
Rank 0 (chunk 0)  →  Rank 1 (chunk 1)  →  Rank 2 (chunk 2)
  ↓                    ↓                    ↓
TimestepNorm:     TimestepNorm:         TimestepNorm:
  count=0            recv count, mean,    recv count, mean,
  mean=0             var from Rank 0      var from Rank 1
  var=init           ↓                    ↓
  ↓                  compute with state   compute with state
  compute            ↓                    ↓
  ↓                  send to Rank 1       send to Rank 2
  send to Rank 1
```

This happens in:
- `TimestepNorm.forward()` (megalodon/modules/timestep_norm.py:106)
- `MovingAverageGatedAttention.forward()` (megalodon/modules/moving_average_gated_attention.py:275)
- `MegaOutputLayer.forward()` (megalodon/model/mega.py:127)

State passing uses `send_to_next_chunk_parallel_region()` and `recv_from_prev_chunk_parallel_region()`.

### Generation (Autoregressive Decoding)

From `megalodon/generation.py:generate()`:

```
1. Encode prompts
   - Tokenize with SentencePiece
   - Truncate to max_prompt_len if needed
  ↓
2. Initialize cache
   - Per-layer: (cache_k, cache_v, count), (prev_count, prev_mean, prev_var), hx
   - Empty at start
  ↓
3. Prefill phase (process chunks of prompt)
   - Process up to MAX_CHUNKS_PER_BATCH chunks at a time
   - Update cache after each batch
   - Truncate cache periodically (keep last TRUNCATE_CHUNKS worth of state)
  ↓
4. Decode phase (generate tokens one at a time)
   for curr_pos in range(start_pos, end_pos):
       logits, cache = model(tokens[:, prev_pos:curr_pos], cache=cache)
       # Sample or argmax
       next_token = sample(logits[:, -1])
       tokens[:, curr_pos] = next_token
       prev_pos = curr_pos
  ↓
5. Return generated tokens
```

Cache structure (per layer):
- `cache_attn`: `(cache_k, cache_v, count)` - KV cache for chunk-local attention
- `cache_norm`: `(prev_count, prev_mean, prev_var)` - TimestepNorm running stats
- `hx`: Complex tensor - CEMA hidden state

## Key Abstractions and Patterns

### 1. Model Parallelism

All linear layers use model-parallel variants from `megalodon/modules/model_parallel/layers.py`:

- **ColumnParallelLinear**: Splits output dimension across GPUs
  - Input: `[B, L, D]` (replicated or parallel)
  - Output: `[B, L, D_local]` (partitioned)
  - Used for: Q/K/V projections, FFN up-projection

- **RowParallelLinear**: Splits input dimension across GPUs
  - Input: `[B, L, D_local]` (partitioned)
  - Output: `[B, L, D]` (all-reduced)
  - Used for: attention output, FFN down-projection

- **ParallelEmbedding**: Splits vocabulary across GPUs
  - Input: token IDs `[B, L]`
  - Output: `[B, L, D_local]`

### 2. FSDP Integration

Model is wrapped with `FullyShardedDataParallel` (fairscale) in `megalodon/model/mega.py:build_model()`:

```python
fsdp_cfg = {
    "process_group": get_data_parallel_group(),
    "model_parallel_process_group": get_model_parallel_group(),
    "chunk_parallel_process_group": get_chunk_parallel_group(),
    "compute_dtype": dtype,
    "mixed_precision": True,
    "flatten_parameters": True,
    "fp32_reduce_scatter": fp32_reduce_scatter,
    "reshard_after_forward": reshard_after_forward,
}
with enable_wrap(wrapper_cls=FullyShardedDataParallel, **fsdp_cfg):
    model = Mega(model_cfg)
    model = wrap(model.cuda())
```

Individual layers are wrapped via `wrap()` calls in the model definition. Optional activation checkpointing via `checkpoint_wrapper`.

### 3. Custom CUDA Kernels

All performance-critical ops have custom CUDA implementations:

| Operation | Python Wrapper | CUDA Files |
|-----------|---------------|------------|
| TimestepNorm | `megalodon/modules/timestep_norm.py` | `csrc/ops/timestep_norm*` |
| EMA hidden state | `megalodon/modules/fused_ops/ema_hidden.py` | `csrc/ops/ema_hidden*` |
| EMA parameters | `megalodon/modules/fused_ops/ema_parameters.py` | `csrc/ops/ema_parameters*` |
| FFT convolution | `megalodon/modules/fused_ops/fftconv.py` | `csrc/ops/fftconv*` |
| Efficient attention | `megalodon/modules/fused_ops/attention/swift.py` | `csrc/ops/attention*` |

Python wrappers define `torch.autograd.Function` classes with `forward()` and `backward()` calling into C++ extensions.

### 4. Configuration System

Dataclass-based configs in `megalodon/config.py`:

- **ModelConf**: Architecture hyperparameters
  - `num_layers`, `model_dim`, `z_dim`, `value_dim`, `num_heads`
  - `chunk_size`, `efficient_attn`, `dropout` rates
  - Predefined configs in `ModelStore` dict (200M, 1.3B, 7.1B, 7.3B)

- **OptimConf**: Optimizer settings (referenced but not used in training)
  - `lr`, `warmup`, `weight_decay`, `beta1`, `beta2`, `clip`

- **ValidConf**: Evaluation settings
  - `ppl_files_str`, `prompt_path`, `batch_size`
  - Generation params: `use_sampling`, `temperature`, `top_k`, `top_p`

### 5. Distributed Process Groups

Three orthogonal parallelism dimensions managed by `megalodon/distributed/initialize.py`:

```python
initialize_model_parallel(model_parallel_size, chunk_parallel_size)
```

Creates three process groups:
- **Data parallel**: Gradients all-reduced across this group (FSDP)
- **Model parallel**: Tensors partitioned across this group (tensor parallelism)
- **Chunk parallel**: Sequence chunks distributed across this group (pipeline-like)

Ranks are organized as:
```
global_rank = data_rank * (model_size * chunk_size) +
              chunk_rank * model_size +
              model_rank
```

Accessed via:
- `get_data_parallel_group()`, `get_data_parallel_rank()`
- `get_model_parallel_group()`, `get_model_parallel_rank()`
- `get_chunk_parallel_group()`, `get_chunk_parallel_rank()`

## Notable Design Choices

1. **Chunk size is fixed at 2048**: Hardcoded in configs, RoPE, and normalization. Changing it requires retraining.

2. **No mixed-chunk processing**: All sequences processed as multiples of chunk_size. Remainder tokens handled separately in inference.

3. **Cache truncation in generation**: Every `TRUNCATE_CHUNKS * chunk_size` tokens, the cache is recomputed from recent context to avoid accumulation errors.

4. **Shared vs separate embeddings**: `share_emb=True` ties input and output embeddings (memory efficient, used in 1.3B pg19 model).

5. **SwiGLU gating**: Optional `swiglu=True` replaces standard FFN with gated variant (SiLU(W1*x) * W3*x).

6. **Efficient attention modes**: `efficient_attn` can be `None` (PyTorch native), `"swift"`, or `"fused"` (both use flash-attention-style kernels).
