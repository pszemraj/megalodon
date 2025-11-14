# Status and Limitations

This document provides an honest assessment of the codebase's maturity, known issues, and missing pieces.

## Maturity by Area

### Stable and Used

**Core Model** (`megalodon/model/mega.py`):
- Status: Production-ready reference implementation
- Well-tested for pretrained checkpoint evaluation
- FSDP integration works reliably
- Model parallel and chunk parallel are functional

**MEGA Attention** (`megalodon/modules/moving_average_gated_attention.py`):
- Status: Stable, implements paper architecture
- CEMA, attention, and gating mechanisms are complete
- Chunk parallel state passing works correctly
- Used in all pretrained models

**Custom CUDA Kernels** (`megalodon/csrc/`):
- Status: Optimized and stable
- TimestepNorm, EMA ops, FFTConv, attention all work
- Backward passes are correct (gradient-checked)
- Performance is good (flash-attention competitive)

**Evaluation Script** (`eval.py`):
- Status: Functional for perplexity and generation
- Multi-GPU evaluation works
- Model parallel and data parallel are stable
- Generation cache management is reliable

**Tokenization** (`megalodon/data/tokenizer.py`):
- Status: Simple SentencePiece wrapper, works as expected
- No issues for standard use cases

### Prototype but Working

**Chunk Parallelism** (`megalodon/modules/chunk_parallel/`):
- Status: Implemented but minimally tested
- State passing (send/recv) works in forward and backward
- **Not tested in eval.py** (only data/model parallel supported)
- Training pseudo-code shows usage but no actual training script
- Likely works but may have edge cases

**CEMA Implementation** (`megalodon/modules/complex_exponential_moving_average.py`):
- Status: Complex but functional
- FFT convolution works correctly
- EMA parameter computation is correct
- **Initialization is opaque**: Random permutation of frequencies synchronized across GPUs via broadcast, but rationale is unclear
- **Caching of coefficients**: In eval mode, coefficients are cached, but no cache invalidation if parameters are modified externally (unlikely in inference)

**DataLoader** (`megalodon/data/dataloader.py`):
- Status: Minimal but functional
- Only supports JSONL with `text` field
- Simple round-robin sharding across ranks
- **No shuffling**: Lines are read in order
- **No error recovery**: Malformed JSON logs error and skips line
- Adequate for evaluation, insufficient for production training

**Distributed Setup** (`megalodon/distributed/`):
- Status: Works for eval.py use case
- Three-way parallelism (data/model/chunk) is correctly initialized
- **No fault tolerance**: Assumes all ranks stay alive
- **No dynamic scaling**: Process group sizes are fixed at init

### Experimental / Incomplete

**Optimizer Building** (referenced in README):
- Status: **Not implemented**
- `build_optimizer()` is mentioned in training pseudo-code but doesn't exist
- Users must implement their own optimizer construction
- Learning rate scheduler is also missing

**Training Loop**:
- Status: **Not implemented**
- Only pseudo-code in README (lines 82-152)
- No actual training script in repo
- No example for data pipeline, checkpointing, logging
- Users must implement from scratch

**Efficient Attention Modes**:
- Status: Partially complete
- `efficient_attn="swift"` and `"fused"` both map to same implementation (`swift_efficient_attention`)
- **"fused" is an alias**: No distinct fused mode (misleading config option)
- `None` mode uses PyTorch native attention (slow, not optimized)
- **Flash-2 integration**: Not present, uses custom flash-1-style kernel

**Generation Cache Truncation** (`generation.py:81-102`):
- Status: Works but hacky
- Every `TRUNCATE_CHUNKS * chunk_size` tokens, cache is recomputed from scratch
- **Magic constants**: `TRUNCATE_CHUNKS=8`, `MAX_CHUNKS_PER_BATCH=2` are hardcoded
- **No truncation strategy**: Could use sliding window or other schemes
- Prevents error accumulation but adds periodic latency spikes

**Checkpoint Reloading** (`megalodon/reloading.py`):
- Status: Works for native checkpoints
- Assumes checkpoint format matches `torch.save()` of model state dict
- **No validation**: Doesn't check config matches checkpoint
- **No version handling**: Assumes checkpoint is compatible
- **No HuggingFace conversion**: Can't load HF models

### Dead / Unused

**Sequence Norm** (`megalodon/csrc/ops/sequence_norm*`):
- Status: CUDA code exists but **never imported** in Python
- Appears to be an alternative to TimestepNorm
- No Python wrapper in `megalodon/modules/`
- Possibly leftover from earlier iteration

**Attention Softmax Kernel** (`megalodon/modules/fused_ops/attention/softmax.py`):
- Wrapper exists but **not used** in model
- `attention_softmax()` function is never called
- Efficient attention uses `swift_efficient_attention` instead
- Unclear if this is legacy or planned alternative

**Config Fields**:
- `ModelConf.output_size`: Defaults to -1 (uses vocab_size), but explicit setting is never used
- `ModelConf.rope_base`: Defaults to None, RoPE uses hardcoded base if None
- `ValidConf.top_k`, `ValidConf.top_p`: Parsed but generation uses `top_k` from separate args (duplication)

**Assertions That Always Pass**:
- `megalodon/config.py:73`: `assert self.z_dim % self.z_dim == 0` (typo, should check `z_dim % num_heads`)
- This means `z_dim` not divisible by `num_heads` will pass assertion but fail later

## Technical Debt and Dragons

### 1. Chunk Size is Hardcoded

**Location**: Throughout codebase (config, RoPE, CEMA, TimestepNorm)

**Issue**: Chunk size (default 2048) is baked into:
- RoPE frequency computation (expects sequences in chunks of 2048)
- TimestepNorm running statistics (reset every chunk)
- CEMA state passing (assumes chunk boundaries)

**Consequence**: Changing `chunk_size` requires retraining from scratch. Can't use pretrained models with different chunk size.

**Dragon**: No runtime check ensures sequences are multiples of chunk size in all code paths. May silently produce wrong results.

### 2. State Passing Gradient Hack

**Location**: `megalodon/modules/moving_average_gated_attention.py:310`, `megalodon/model/mega.py:149`

**Code**:
```python
prev_tensor = send_to_next_chunk_parallel_region(prev_tensor)
# TODO: more elegent solution to force call bwd of prev_tensor
out_cema = out_cema + prev_tensor.to(out_cema).mean() * 0
```

**Issue**: To ensure gradients flow through chunk-parallel communication, the sent tensor is added to the output with coefficient 0. This forces autograd to include it in the computation graph.

**Dragon**: Fragile. If PyTorch's autograd optimizations become smarter and prune this no-op, gradients will break.

**Better solution**: Use `torch.autograd.Function` to explicitly register communication as part of the computation graph.

### 3. Model Parallel Assumptions

**Location**: `megalodon/modules/model_parallel/layers.py`

**Assumptions**:
- All parallel dimensions (model_dim, num_heads, vocab_size) must be **exactly divisible** by `model_parallel_size`
- No handling for odd sizes (e.g., vocab_size = 50257 with 8-way parallel fails)

**Dragon**: Silent failure. If not divisible, `divide_and_check_no_remainder()` asserts, crashing distributed init. Error message is cryptic:
```
AssertionError: 50257 is not divisible by 8
```

**Workaround**: Pad vocabulary or choose compatible model_parallel_size. No automatic padding.

### 4. Mixed Dtype Handling

**Location**: Throughout model, especially `megalodon/model/mega.py:242`

**Code**:
```python
out = self.output(x, cache_output)
return out.float(), cache
```

**Issue**: Model computes in BF16/FP16 but forces output to FP32. This is **always** done, even when BF16 output is acceptable (e.g., for loss computation).

**Consequence**: Extra memory and compute for dtype conversion. Also, cache tensors remain in compute dtype, leading to mixed-dtype caches.

**Dragon**: If you try to keep everything in BF16 (e.g., for performance), the `.float()` call breaks that.

### 5. Evaluation Dummy Examples

**Location**: `eval.py:81-85`

**Code**:
```python
len_data = reduce_scalar(len(batches), op='max')
extra = int(len_data) - len(batches)
if extra > 0:
    batches.extend([batches[0] for _ in range(extra)])
```

**Issue**: To ensure all ranks process the same number of batches (required for FSDP), ranks with fewer examples **duplicate the first batch** to pad.

**Dragon**: If you're not careful, these dummy examples are included in metrics. The code correctly excludes them via `dummy_start` check, but if you modify the eval loop, it's easy to forget and count duplicates.

### 6. Global State in CEMA Initialization

**Location**: `megalodon/modules/complex_exponential_moving_average.py:25`

**Code**:
```python
idx = torch.randperm(embed_dim)
idx_ = idx.cuda()
dist.broadcast(idx_, src=0)
```

**Issue**: Random permutation is generated on CPU, moved to GPU, broadcasted from rank 0, then used to initialize parameters.

**Dragons**:
- Assumes rank 0 exists and is alive at init time
- Assumes `dist` is initialized before model construction
- If you reinitialize a model, the permutation is **different** (not seeded deterministically)
- This breaks checkpoint loading if you rebuild a model from config

**Workaround**: Always load pretrained checkpoints. Don't rebuild models from config and expect same initialization.

### 7. Cache Structure is Brittle

**Location**: `megalodon/model/mega.py:192-240`, `megalodon/generation.py:72`

**Structure**:
```python
cache = (
    [layer_cache_1, layer_cache_2, ...],  # Per-layer caches
    output_cache                          # Output layer cache
)
layer_cache = (
    cache_attn,   # (cache_k, cache_v, count)
    cache_norm,   # (prev_count, prev_mean, prev_var)
    hx            # Complex tensor
)
```

**Issue**: Cache is a nested tuple structure with implicit ordering. No named fields, no type checking.

**Dragons**:
- Easy to unpack in wrong order
- If you add a new cached component, must update all cache-handling code
- No validation that cache matches model config (e.g., num_layers)

**Better solution**: Use dataclass or NamedTuple for cache structure.

### 8. Fire Argument Parsing

**Location**: `eval.py:160-197`

**Issue**: Uses `fire` library for CLI parsing, which:
- Converts all args to strings initially
- Requires manual parsing for bools (`parse_bool_flag`), ints, floats
- No type validation until runtime
- No help text or argument descriptions

**Dragon**: Typos in arg names silently become new unexpected kwargs. E.g., `--model_parrallel_size` (typo) will be accepted but not used, defaulting to None and likely crashing later.

### 9. Tokenizer Assumptions

**Location**: `megalodon/data/tokenizer.py`, `megalodon/generation.py`

**Assumptions**:
- Tokenizer has `.bos_id`, `.eos_id`, `.pad_id` attributes
- These are single token IDs (not lists)
- EOS token is single token (generation uses `cut_at_eos=True`)

**Dragon**: If you use a tokenizer without these attributes (e.g., some HuggingFace tokenizers), code crashes with AttributeError. No fallback or error message.

### 10. FFT Padding

**Location**: `megalodon/modules/fused_ops/fftconv.py`

**Issue**: FFT convolution pads sequence to next power of 2 for efficiency. For length 2049, pads to 4096 (nearly 2x).

**Consequence**: Wastes computation and memory for sequences just over a power of 2.

**Optimization**: Could use more flexible FFT sizes (e.g., products of small primes), but cuFFT works best with powers of 2.

**Dragon**: Very inefficient for lengths like 2049, 4097, 8193. Prefer lengths that are powers of 2 or just under.

## Missing Pieces

### 1. Training Script

**What's missing**:
- Data pipeline (sharding, shuffling, preprocessing)
- Optimizer construction and scheduling
- Gradient accumulation
- Checkpointing (saving and loading during training)
- Logging (loss, metrics, throughput)
- Validation loop
- Early stopping or convergence criteria

**Workaround**: Implement based on pseudo-code in README. Use `build_model()` and define your own training loop.

### 2. Learning Rate Scheduler

**What's missing**:
- `build_optimizer()` function referenced in pseudo-code
- No implementations of schedulers (linear, cosine, inverse sqrt, etc.)
- `OptimConf.scheduler` field exists but unused

**Workaround**: Use PyTorch's `torch.optim.lr_scheduler` or implement custom scheduler.

### 3. Data Preprocessing

**What's missing**:
- Tokenization scripts
- Data cleaning, deduplication
- JSONL conversion from common formats (text, parquet, arrow)
- Train/val splits

**Workaround**: Prepare your own JSONL data. Use `tokenizer.encode()` if you need to pre-tokenize.

### 4. Evaluation Benchmarks

**What's missing**:
- MMLU, HellaSwag, ARC, etc. integration
- Zero-shot / few-shot evaluation
- Prompt templates for tasks
- Metric computation (accuracy, F1, etc.)

**Workaround**: Implement yourself or use external harness (e.g., EleutherAI's lm-evaluation-harness, but requires adaptation).

### 5. Checkpoint Conversion

**What's missing**:
- HuggingFace model conversion (to/from)
- GGUF export for llama.cpp
- Quantization (INT8, INT4, etc.)

**Workaround**: Implement custom conversion scripts. Model architecture is different enough from standard transformers that automatic conversion is unlikely to work.

### 6. Serving / Deployment

**What's missing**:
- Inference server (e.g., vLLM, TGI integration)
- Batching for throughput
- Speculative decoding
- KV cache optimization

**Workaround**: Use `generation.py:generate()` as-is for single-batch inference. Implement batching yourself if needed.

### 7. Unit Tests

**What's missing**:
- No `tests/` directory
- No pytest or unittest files
- No CI/CD

**Consequence**: Changes may introduce regressions. Gradient checks are not automated.

**Workaround**: Write your own tests if modifying the code.

### 8. Documentation for Custom Kernels

**What's missing**:
- No inline comments in CUDA code explaining algorithms
- No docs for kernel APIs (input/output shapes, constraints)
- No performance benchmarks or comparisons

**Workaround**: Read CUDA code carefully. Use Python wrappers as API documentation.

### 9. Model Compression

**What's missing**:
- No pruning, distillation, or quantization
- No mixed-precision training beyond BF16/FP16

**Workaround**: Implement yourself. CEMA's complex numbers may complicate quantization.

### 10. Multi-Node Training

**What's missing**:
- No SLURM integration (referenced in README but not implemented)
- No example multi-node launch scripts
- No cross-node performance tuning

**Workaround**: Use `torchrun` with `--nproc_per_node` and `--nnodes` flags. Ensure fast interconnect (InfiniBand) for acceptable performance.

## Known Scaling Issues

### Long Context Stability

**Issue**: Very long sequences (>100k tokens) may have numerical stability issues in TimestepNorm running statistics.

**Symptom**: Diverging loss, NaN values after many chunks.

**Mitigation**: Cache truncation in generation helps. For training, ensure `fp32_reduce_scatter=True` in FSDP.

### Memory Scaling

**Issue**: Generation cache grows linearly with sequence length. For 100k tokens, cache can be tens of GB.

**Mitigation**: Cache truncation is enabled by default. Can also reduce `TRUNCATE_CHUNKS` to truncate more aggressively.

### Chunk Parallel Scaling

**Issue**: Chunk parallel has sequential dependency (each chunk waits for previous). Limited speedup from parallelism.

**Observed**: 2x chunk parallel gives <1.5x speedup in practice (communication overhead).

**Mitigation**: Use chunk parallel only when data parallel is exhausted (very large batches).

## Experimental Features to Avoid

1. **Chunk parallel in evaluation**: Disabled in eval.py for a reason. Likely has bugs.
2. **Prior count in TimestepNorm**: Setting `prior_count > 0` without careful tuning may hurt performance.
3. **Changing chunk_size**: Don't do this unless retraining from scratch.
4. **LayerWise checkpointing with FSDP**: May have compatibility issues (fairscale FSDP + checkpoint_wrapper is finicky).
5. **Mixed model_parallel and chunk_parallel**: Not well-tested. Stick to one or the other.

## Recommendations for Production Use

If you're using this in production:

1. **Stick to evaluation mode**: Training is experimental (no script provided).
2. **Use pretrained checkpoints**: Don't try to train from scratch without significant effort.
3. **Model parallel only**: Avoid chunk parallel until you've thoroughly tested it.
4. **Pin dependencies**: `fairscale` is unmaintained, pin to exact commit to avoid breakage.
5. **Test thoroughly**: No unit tests means you're on your own for validation.
6. **Monitor for NaNs**: Long-context runs may have numerical issues.
7. **Profile first**: Custom kernels are optimized but may have edge cases where PyTorch is faster.

## Future Work (Not in Repo)

Features that would improve the codebase but aren't present:

- Gradient checkpointing integration with chunk parallel
- Flash-2 attention kernel
- INT8/FP8 quantization for deployment
- Better cache data structures (e.g., Paged attention)
- Multi-query attention (MQA) or grouped-query attention (GQA)
- Proper training script with all the bells and whistles
- HuggingFace Transformers integration
- Comprehensive unit and integration tests
- Better error messages and validation
