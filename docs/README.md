# Megalodon: What This Project Is

## Overview

Megalodon is a reference implementation of a novel large language model architecture designed to handle **unlimited context lengths** efficiently during both pretraining and inference. Unlike standard Transformer-based LLMs that face quadratic complexity with sequence length, Megalodon achieves efficient long-context processing through a hybrid architecture combining:

- **Moving Average Gated Attention (MEGA)**: A custom attention mechanism that processes sequences in fixed-size chunks
- **Complex Exponential Moving Average (CEMA)**: Linear-complexity sequence modeling that replaces full self-attention for long-range dependencies
- **Chunk-parallel processing**: Enables training and inference on arbitrarily long sequences by splitting them into manageable chunks

This implementation targets large-scale pretraining at the 7B parameter scale, with pretrained checkpoints available for evaluation.

## What's Actually Here

This repository provides:

- **Core model implementation** (`megalodon/model/mega.py`):
  - MegaBlock layer combining MEGA attention and normalized feedforward networks
  - Support for model sizes from 200M to 7.3B parameters (configs in `ModelStore`)
  - Chunk-based sequence processing with configurable chunk size (default: 2048 tokens)

- **Custom CUDA kernels** (`megalodon/csrc/`):
  - TimestepNorm: running normalization across sequence timesteps
  - EMA operations: efficient hidden state and parameter computation for CEMA
  - FFTConv: FFT-based convolution for sequence processing
  - Efficient attention: fused attention implementations (swift, fused variants)

- **Distributed training support** (`megalodon/distributed/`):
  - FSDP (Fully Sharded Data Parallel) integration via fairscale
  - Model parallelism (tensor parallelism across GPUs)
  - Chunk parallelism (pipeline-style parallelism for long sequences)
  - Three orthogonal parallelism dimensions: data, model, and chunk

- **Evaluation script** (`eval.py`):
  - Perplexity evaluation on JSONL datasets
  - Text generation with sampling or greedy decoding
  - Supports multi-GPU evaluation with model and data parallelism

- **Inference utilities** (`megalodon/generation.py`, `megalodon/inference.py`):
  - Autoregressive generation with KV-like caching (adapted for MEGA architecture)
  - Cache truncation for extremely long generations
  - Batch processing with dynamic prompt lengths

## What's Notably Not Here

- **No training script**: Only pseudo-code is provided in the README. You must implement your own training loop, data pipeline, and optimization schedule.

- **No data preprocessing**: The repo assumes you have data in JSONL format with a `text` field. No tokenization, cleaning, or deduplication tools are provided.

- **No evaluation harness**: Perplexity and basic generation are supported, but no integration with benchmarks like MMLU, HellaSwag, or other LLM evaluation suites.

- **No deployment code**: No serving infrastructure, quantization, or inference optimization beyond what's in the core model.

- **No checkpoint conversion**: Assumes you use the native checkpoint format. No HuggingFace model conversion utilities.

- **No optimizer implementation**: Uses standard PyTorch optimizers. The pseudo-code references a custom `build_optimizer` function that doesn't exist in the repo.

## Why It Matters

Megalodon diverges from standard Transformers in several critical ways:

1. **Linear complexity for long sequences**: CEMA processes sequences with O(n) complexity instead of O(n²), enabling training and inference on sequences far longer than typical transformer context windows.

2. **Chunk-based architecture**: The model processes sequences in fixed-size chunks (2048 tokens by default), with state passed between chunks. This enables:
   - Training on infinite-length sequences via chunk parallelism
   - Inference with bounded memory regardless of sequence length
   - Efficient caching that resets every chunk

3. **Custom normalization**: TimestepNorm maintains running statistics across timesteps within a chunk, essential for the chunk-parallel training regime.

4. **Specialized attention**: The MEGA attention mechanism combines:
   - CEMA for long-range dependencies
   - Chunk-local attention for within-chunk interactions
   - Rotary positional embeddings applied per-chunk

5. **Three-way parallelism**: Unlike standard data/model parallelism, Megalodon adds chunk parallelism, allowing different GPUs to process different chunks of the same sequence in a pipeline-parallel fashion.

This architecture is designed for scenarios where context length is critical (long document understanding, code with large repositories, etc.) and where standard transformers become prohibitively expensive.
