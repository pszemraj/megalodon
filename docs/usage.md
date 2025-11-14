# Usage

This document covers practical usage of the Megalodon model for evaluation and inference.

## Prerequisites

- Installed environment (see `docs/install.md`)
- Pretrained checkpoint directory with:
  - Model checkpoint file(s)
  - `params.json` config file
  - Tokenizer model (SentencePiece `.model` file)
- Data in JSONL format (one JSON per line with `"text"` field)

## Evaluation

### Perplexity Evaluation

Evaluate a model's perplexity on one or more JSONL files:

```bash
export NGPU=4
torchrun --nproc_per_node=$NGPU eval.py \
    --checkpoint_dir /path/to/checkpoint \
    --dump_dir ./eval_results \
    --dtype bf16 \
    --model_parallel_size 1 \
    --ppl_files_str "data/eval1.jsonl,data/eval2.jsonl" \
    --batch_size 2
```

**Parameters**:
- `checkpoint_dir`: Directory containing checkpoint and `params.json`
- `dump_dir`: Where to save evaluation outputs
- `dtype`: `bf16`, `fp16`, or `fp32` (default: `bf16`)
- `model_parallel_size`: Number of GPUs for tensor parallelism (must divide `NGPU`)
- `ppl_files_str`: Comma-separated list of JSONL files
- `batch_size`: Batch size per GPU
- `tokenizer_path`: (Optional) Override tokenizer from checkpoint

**JSONL Format**:
```json
{"text": "The quick brown fox jumps over the lazy dog."}
{"text": "Another example sentence for evaluation."}
```

**Output**:
Logs will show perplexity and token count for each file:
```
PPL on data/eval1.jsonl: 12.34, w. 1000000 tokens.
```

### Text Generation

Generate text from prompts:

```bash
torchrun --nproc_per_node=4 eval.py \
    --checkpoint_dir /path/to/checkpoint \
    --dump_dir ./generations \
    --prompt_path prompts.jsonl \
    --gen_seq_len 512 \
    --use_sampling true \
    --temperature 0.8 \
    --batch_size 1
```

**Parameters**:
- `prompt_path`: JSONL file with prompts (same format as PPL data)
- `gen_seq_len`: Maximum tokens to generate per prompt (default: 512)
- `use_sampling`: `true` for sampling, `false` for greedy (default: false)
- `temperature`: Sampling temperature (default: 1.0)
- `top_k`: Top-k sampling (default: 0 = disabled)
- `top_p`: Nucleus sampling (default: 0.0 = disabled)

**Output**:
Generations saved to `{dump_dir}/generations/{rank}.jsonl`:
```json
{"prompt": "Once upon a time", "generation": " there was a brave knight..."}
```

### Combined Evaluation

Run both PPL and generation in one command:

```bash
torchrun --nproc_per_node=4 eval.py \
    --checkpoint_dir /path/to/checkpoint \
    --dump_dir ./results \
    --ppl_files_str "eval.jsonl" \
    --prompt_path "prompts.jsonl" \
    --batch_size 2
```

The script runs PPL first, then generation.

## Programmatic Usage

### Loading a Model

```python
from megalodon.reloading import reload_model

model, tokenizer, config = reload_model(
    checkpoint_dir="/path/to/checkpoint",
    model_parallel_size=1,
    chunk_parallel_size=1,
    dtype="bf16",
    tokenizer_path=None,  # Uses checkpoint tokenizer if None
)

model.eval()  # Set to eval mode
```

**Returns**:
- `model`: `Mega` instance wrapped in FSDP
- `tokenizer`: `Tokenizer` instance (SentencePiece wrapper)
- `config`: `ModelConf` dataclass with architecture params

**Requirements**:
- Must call after `torch.distributed.init_process_group()`
- Must call `initialize_model_parallel()` before loading

### Inference (Perplexity)

```python
import torch
from megalodon.inference import inference

# Prepare data
tokens = torch.tensor([[1, 2, 3, 4, 5]]).cuda()  # [batch, seq_len]
targets = torch.tensor([[2, 3, 4, 5, 6]]).cuda()  # Shifted by 1

# Run inference
with torch.inference_mode():
    loss = inference(model, tokens, targets, dtype="bf16")
    # loss: [batch, seq_len] per-token cross-entropy

ppl = torch.exp(loss.mean())
print(f"Perplexity: {ppl.item()}")
```

**Notes**:
- `targets` should have `-100` for positions to ignore (padding)
- Sequences are automatically padded to chunk boundaries
- Handles multi-chunk sequences with caching

### Generation (Autoregressive)

```python
from megalodon.generation import generate

prompts = ["Once upon a time", "In the beginning"]

with torch.inference_mode():
    generated = generate(
        model=model,
        dtype="bf16",
        tokenizer=tokenizer,
        prompt=prompts,
        max_gen_len=256,
        use_sampling=True,
        temp=0.8,
        top_p=0.9,
        remove_prompts=True,
    )

# generated: List[List[int]] - token IDs
for tokens in generated:
    text = tokenizer.decode(tokens, cut_at_eos=True)
    print(text)
```

**Parameters**:
- `prompt`: List of strings, or `None` for padding-only (multi-GPU sync)
- `max_prompt_len`: Truncate prompts to this length (default: 256)
- `max_gen_len`: Generate this many tokens (default: 256)
- `use_sampling`: Sample vs greedy decoding
- `temp`: Temperature for sampling
- `top_k`, `top_p`: Sampling strategies (0 = disabled)
- `remove_prompts`: Return only generated tokens (not input)
- `truncate_chunks`: Advanced - truncate cache for very long prompts

**Constraints**:
- All prompts in a batch must fit in memory
- Very long prompts (>8 chunks = 16k tokens) may hit cache limits
- Single-batch generation only (no multi-batch prefill)

### Building a Model from Scratch

```python
from megalodon.model.mega import build_model, ModelConf
from megalodon.distributed import initialize_model_parallel

# Setup distributed
initialize_model_parallel(model_parallel_size=1, chunk_parallel_size=1)

# Define config
cfg = ModelConf(
    num_layers=24,
    model_dim=2048,
    z_dim=512,
    value_dim=4096,
    num_heads=2,
    ffn_hidden_dim=4864,
    chunk_size=2048,
    vocab_size=50000,
    efficient_attn="swift",
)

# Build model
model = build_model(
    model_cfg=cfg,
    dtype="bf16",
    fp32_reduce_scatter=False,
    reshard_after_forward=True,
)

model.train()  # Ready for training
```

**Config Options**:
- `num_layers`: Transformer layers (e.g., 12, 24, 32)
- `model_dim`: Hidden dimension (e.g., 1024, 2048, 4096)
- `z_dim`: Attention query/key dimension (must be divisible by `num_heads`)
- `value_dim`: Attention value dimension (must be divisible by `num_heads`)
- `num_heads`: Number of attention heads
- `ffn_hidden_dim`: FFN intermediate dimension
- `cema_ndim`: CEMA dimensionality (default: 16)
- `chunk_size`: Fixed chunk size (default: 2048)
- `vocab_size`: Vocabulary size (required)
- `efficient_attn`: `None`, `"swift"`, or `"fused"`
- `dropout`, `attention_dropout`, `hidden_dropout`: Dropout rates
- `swiglu`: Use SwiGLU FFN (default: False)
- `scale_emb`: Scale embeddings by sqrt(model_dim) (default: False)
- `share_emb`: Tie input/output embeddings (default: False)

**Predefined Configs**:
```python
from megalodon.model.mega import ModelStore

cfg = ModelStore["mega7.3B"]  # 7.3B parameter model
cfg.vocab_size = 50000  # Must set vocab_size
```

Available: `"mega200M"`, `"mega1.3B"`, `"mega1.3B_pg19"`, `"mega7.1B"`, `"mega7.3B"`.

## Data Preparation

The repo expects JSONL format with a `"text"` field:

```json
{"text": "First document text here."}
{"text": "Second document text here."}
```

**For Perplexity**:
- Each line is one evaluation example
- No length requirements (DataLoader handles padding)
- Use `batch_size` to control memory

**For Generation**:
- Each line is one prompt
- Prompts longer than `max_prompt_len` are truncated on the left
- All prompts in a batch are processed together

**Data Sharding**:
- `eval.py` automatically shards data across data-parallel ranks
- Rank `i` processes lines where `(line_num - 1) % world_size == i`

## Configuration and Customization

### Changing Model Size

Edit `ModelConf` or use predefined configs:

```python
# Start from preset
cfg = ModelStore["mega1.3B"]
# Modify
cfg.num_layers = 16  # Reduce layers
cfg.vocab_size = 32000  # Set vocab
```

### Changing Chunk Size

**WARNING**: Chunk size is baked into the architecture:
- RoPE frequencies are computed per-chunk
- TimestepNorm and CEMA expect fixed chunk size
- Changing `chunk_size` requires retraining from scratch

To change for a new model:
```python
cfg = ModelConf(chunk_size=4096, ...)  # Must retrain
```

### Changing Attention Type

```python
cfg.efficient_attn = "swift"  # or "fused" or None
```

- `None`: PyTorch native attention (slow, supports masking)
- `"swift"`: Flash-attention-style fused kernel (fast)
- `"fused"`: Alias for `"swift"` (same implementation)

### Enabling SwiGLU

```python
cfg.swiglu = True  # Use gated FFN
```

Increases FFN FLOPs by ~33% but often improves quality.

### Dropout Rates

```python
cfg.dropout = 0.1            # Post-residual dropout
cfg.attention_dropout = 0.1  # Attention scores dropout
cfg.hidden_dropout = 0.1     # Intermediate activations dropout
```

### Model Parallelism

Split model across GPUs:

```bash
# 8 GPUs total, 4-way model parallel
torchrun --nproc_per_node=8 eval.py \
    --model_parallel_size 4 \
    ...
```

Requirements:
- `num_heads` must be divisible by `model_parallel_size`
- `model_dim` must be divisible by `model_parallel_size`
- Embedding vocabulary is sharded across ranks

### Chunk Parallelism

**Only for training** (not supported in eval.py):

```python
initialize_model_parallel(model_parallel_size=2, chunk_parallel_size=2)
```

Sequences are split into chunks, with different GPUs processing different chunks in pipeline fashion.

## Limits and Edge Cases

### Sequence Length Constraints

**Training/Evaluation**:
- Total sequence length must be divisible by `chunk_parallel_size`
- Each chunk must be divisible by `chunk_size` (2048 by default)
- Example: 8192 tokens → 4 chunks → can use `chunk_parallel_size` up to 4

**Generation**:
- `chunk_parallel_size` must be 1 (generation doesn't support chunk parallel)
- Prompts can be any length (handled via chunking internally)
- Very long prompts (>16k tokens) trigger cache truncation

### Memory Limits

**Batch Size**:
- Each example in batch must fit in GPU memory
- Long sequences (e.g., 32k tokens) require small batch sizes
- Use gradient accumulation for effective larger batches (implement yourself)

**Generation Cache**:
- Cache grows with sequence length
- Automatic truncation every `TRUNCATE_CHUNKS * chunk_size` tokens (default: 16k)
- Can override with `truncate_chunks` parameter in `generate()`

### Tokenizer Constraints

- Must be SentencePiece `.model` file
- BOS token ID assumed to be `tokenizer.bos_id`
- EOS token ID assumed to be `tokenizer.eos_id`
- PAD token ID assumed to be `tokenizer.pad_id`

### Multi-GPU Sync

All ranks must:
- Process the same number of batches (eval.py pads with dummy examples)
- Call model the same number of times (generation pads prompts with `FAKE_PROMPT`)
- Use the same random seed for sampling (set via `--seed`)

## Troubleshooting

**"chunk parallel does not support evaluation"**:
- Evaluation requires `chunk_parallel_size=1`
- Set explicitly in `reload_model()` or ensure single-GPU eval

**"sequence length not divisible by chunk_size"**:
- Ensure data is padded to multiples of 2048 tokens
- Dataloader should handle this automatically for evaluation

**"tensor size mismatch" during model load**:
- `model_parallel_size` must match checkpoint's model parallel size
- Check `params.json` in checkpoint for correct size

**Out of memory during generation**:
- Reduce `batch_size`
- Use shorter `max_gen_len`
- Enable `truncate_chunks` for very long prompts

**Perplexity is NaN**:
- Check data quality (no empty strings, valid Unicode)
- Verify tokenizer is correct for checkpoint
- Try `dtype="fp32"` to rule out numerical issues
