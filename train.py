#!/usr/bin/env python
"""train.py: Train a Megalodon model using distributed training infrastructure, with validation and generation checks."""

import logging
import random
from pathlib import Path

import fire
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from megalodon.config import OptimConf, TokenizerConf
from megalodon.data.dataloader import DataLoader
from megalodon.data.tokenizer import Tokenizer
from megalodon.distributed import (
    get_chunk_parallel_world_size,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_model_parallel_rank,
    get_model_parallel_world_size,
    init_signal_handler,
    init_torch_distributed,
    initialize_model_parallel,
)
from megalodon.logger import add_logger_file_handler, initialize_logger
from megalodon.model.mega import ModelStore, build_model
from megalodon.modules.losses import cross_entropy
from megalodon.optim import build_optimizer
from megalodon.utils import (
    check_ampere_gpu,
    clip_grad_norm_,
    log_host,
    set_random_seed,
    setup_env,
)

logger = logging.getLogger()


def save_checkpoint(
    model, optimizer, scheduler, step: int, output_dir: Path, is_final: bool = False
):
    """Save model checkpoint from main process."""
    if get_data_parallel_rank() != 0:
        return

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "model_final.pt" if is_final else f"checkpoint-{step}.pt"
    torch.save(checkpoint, output_dir / filename)
    logger.info(f"Saved checkpoint at step {step}")


@torch.no_grad()
def run_validation(model, val_loader, max_batches: int = 10):
    """
    Run validation to estimate validation loss.
    We'll take up to `max_batches` batches from val_loader to compute avg loss.
    """
    model.eval()
    total_loss = 0
    count = 0
    for _ in range(max_batches):
        try:
            batch = next(val_loader)
        except StopIteration:
            break
        pred, _ = model(batch["x"].cuda())
        loss = cross_entropy(pred, batch["y"].cuda()).mean().item()
        total_loss += loss
        count += 1
    model.train()
    return total_loss / max(count, 1)


@torch.no_grad()
def top_k_sampling(logits, top_k=10):
    """
    Perform top-k sampling from the logits.
    """
    if top_k > 0:
        # Get top K indices
        values, indices = torch.topk(logits, k=top_k, dim=-1)
        # Filter out everything not in top k
        filtered_logits = torch.full_like(logits, float("-inf"))
        filtered_logits.scatter_(1, indices, values)
        logits = filtered_logits
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_from_prompt(model, tokenizer, prompt_tokens, gen_length=100, top_k=10):
    """
    Generate text from a given prompt tokens using top-k sampling.
    """
    model.eval()
    x = torch.tensor(prompt_tokens, dtype=torch.long, device="cuda").unsqueeze(0)
    for _ in range(gen_length):
        pred, _ = model(x)
        # Take last token logits
        logits = pred[:, -1, :]
        next_token = top_k_sampling(logits, top_k=top_k)
        x = torch.cat([x, next_token], dim=1)
    # Decode
    out_seq = x[0].tolist()
    text = tokenizer.decode(out_seq, cut_at_eos=False)
    model.train()
    return text


def train(
    data_dir: str,
    tokenizer_path: str,
    model_name: str = "mega200M",
    output_dir: str = None,
    batch_size: int = 4,
    grad_acc_steps: int = 8,
    max_seq_length: int = None,
    max_steps: int = 100000,
    save_steps: int = 5000,
    validation_steps: int = 1000,
    generation_steps: int = 2000,
    gen_length: int = 100,
    top_k: int = 10,
    learning_rate: float = 1e-4,
    warmup_steps: int = 500,
    model_parallel_size: int = 1,
    chunk_parallel_size: int = 1,
    seed: int = 42,
    dtype: str = "fp32",
    autocast_enabled: bool = True,
    fp32_reduce_scatter: bool = False,
    reshard_after_forward: bool = True,
    distributed_timeout: int = 1800,  # 30 minutes
    log_every_n_steps: int = 10,
):
    """
    Train a Megalodon model using distributed infrastructure, with validation & generation checks.
    Launch with: torchrun --nproc_per_node={NUM_GPUS} train.py [args]

    Args:
        data_dir: Directory with training data files: train.jsonl and optionally validation.jsonl
        tokenizer_path: Path to SentencePiece tokenizer
        model_name: Name of model configuration
        output_dir: Directory to save checkpoints
        batch_size: Batch size per GPU
        grad_acc_steps: Gradient accumulation steps
        max_seq_length: Maximum sequence length
        max_steps: Maximum training steps
        save_steps: Save checkpoint every N steps
        validation_steps: Run validation every N steps if validation set exists
        generation_steps: Run generation sanity check every N steps if validation set exists
        gen_length: Length of generation for sanity check
        top_k: Top-k for sampling during generation
        learning_rate: Learning rate
        warmup_steps: Learning rate warmup steps
        model_parallel_size: Model parallel size
        chunk_parallel_size: Chunk parallel size
        seed: Random seed
        dtype: Model dtype (bf16/fp16/fp32)
        fp32_reduce_scatter: Use FP32 for reduce scatter
        reshard_after_forward: Reshard parameters after forward pass
        distributed_timeout: Timeout for distributed operations
    """
    # Basic setup
    initialize_logger()
    setup_env()
    log_host()
    set_random_seed(seed)
    check_ampere_gpu()

    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path.cwd() / "checkpoints" / model_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    add_logger_file_handler(output_dir / "train.log")

    # Initialize distributed training
    init_signal_handler()
    logger.info("Initializing distributed training...")
    is_slurm, global_rank, world_size = init_torch_distributed(
        timeout=distributed_timeout
    )

    logger.info("Initializing model parallel...")
    initialize_model_parallel(model_parallel_size, chunk_parallel_size)

    # Log distributed training setup
    logger.info(
        f"Distributed setup - "
        f"Global rank: {global_rank}/{world_size} | "
        f"Model parallel: {get_model_parallel_rank()}/{get_model_parallel_world_size()} | "
        f"Data parallel: {get_data_parallel_rank()}/{get_data_parallel_world_size()} | "
        f"Chunk parallel: {get_chunk_parallel_world_size()}"
    )

    # Load configurations
    model_cfg = ModelStore[model_name]
    tokenizer_cfg = TokenizerConf(path=tokenizer_path)
    optim_cfg = OptimConf(lr=learning_rate, warmup=warmup_steps)

    # Initialize tokenizer
    tokenizer = Tokenizer(tokenizer_cfg)
    model_cfg.vocab_size = model_cfg.output_size = tokenizer.sp_model_vocab_size

    # Train dataloader
    train_file = Path(data_dir) / "train.jsonl"
    train_loader = DataLoader(
        tokenizer=tokenizer,
        path=str(train_file),
        batch_size=batch_size,
        world_rank=global_rank,
        world_size=world_size,
        chunk_size=model_cfg.chunk_size,
        max_seq_length=max_seq_length,
        shuffle=True,
    )
    train_iter = iter(train_loader)

    # Validation dataloader (optional)
    val_file = Path(data_dir) / "validation.jsonl"
    val_loader = None
    if val_file.is_file():
        # If validation exists, load it
        val_loader_obj = DataLoader(
            tokenizer=tokenizer,
            path=str(val_file),
            batch_size=batch_size,
            world_rank=global_rank,
            world_size=world_size,
            chunk_size=model_cfg.chunk_size,
            max_seq_length=max_seq_length,
        )
        val_iter = iter(val_loader_obj)
        # We'll re-init the val_iter each time we run validation
        # or just continuously iterate
        # For validation and generation prompt extraction, we can re-instantiate when needed
        logger.info(
            "Validation file found. Will run periodic validation and generation checks."
        )
        val_loader = val_loader_obj
    else:
        logger.info(
            "No validation file found. Skipping validation and generation steps."
        )

    # Build model
    logger.info("Building model...")
    model = build_model(
        model_cfg,
        dtype=dtype,
        fp32_reduce_scatter=fp32_reduce_scatter,
        reshard_after_forward=reshard_after_forward,
    )
    logger.info(str(model_cfg))

    # Initialize optimizer and scheduler
    optimizer, scheduler = build_optimizer(
        model=model, cfg=optim_cfg, total_steps=max_steps, param_dtype=dtype
    )

    # Training loop
    logger.info("Starting training...")
    model.train()
    step = 0
    optimizer.zero_grad()
    accumulated_loss = 0
    pbar = tqdm(total=max_steps, mininterval=10, desc="Training")

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            # Reinit train_iter if needed
            train_iter = iter(train_loader)
            batch = next(train_iter)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            pred, _ = model(batch["x"].cuda())
            loss = cross_entropy(pred, batch["y"].cuda()).mean()
        loss_value = loss.item()

        # Backward pass with gradient accumulation
        (loss / grad_acc_steps).backward()
        accumulated_loss += loss_value

        if (step + 1) % grad_acc_steps == 0:
            # Gradient sync and clip
            model.grad_all_reduce()
            clip_grad_norm_(fsdp_module=model, max_norm=optim_cfg.clip)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            avg_loss = accumulated_loss / grad_acc_steps
            if step % log_every_n_steps == 0:
                logger.info(
                    f"Step {step} | Loss: {avg_loss:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )
            accumulated_loss = 0

        # Validation step
        if val_loader is not None and (step > 0) and (step % validation_steps == 0):
            logger.info("Running validation...")
            val_iter = iter(val_loader)
            val_loss = run_validation(model, val_iter, max_batches=10)
            logger.info(f"Step {step} | Validation Loss: {val_loss:.4f}")

        # Generation step
        if val_loader is not None and (step > 0) and (step % generation_steps == 0):
            # We'll pick a random line from validation set to use as prompt
            # For simplicity, re-iterate val_loader_obj into memory for a single example
            val_iter = iter(val_loader)
            val_batch = next(val_iter)
            # pick a random sequence from val_batch
            x_tokens = val_batch["x"][0].tolist()
            # pick a random start index for a small prompt
            prompt_length = min(50, len(x_tokens))
            start_idx = random.randint(0, max(0, len(x_tokens) - prompt_length - 1))
            prompt_tokens = x_tokens[start_idx : start_idx + prompt_length]

            prompt_text = tokenizer.decode(prompt_tokens, cut_at_eos=False)
            logger.info(f"\n\n[Generation Step] Prompt:\n{prompt_text}\n{'='*50}")
            generated_text = generate_from_prompt(
                model, tokenizer, prompt_tokens, gen_length=gen_length, top_k=top_k
            )
            logger.info(
                f"[Generation Step] Generated continuation:\n{generated_text}\n{'='*50}"
            )

        # Checkpointing
        if step > 0 and step % save_steps == 0:
            save_checkpoint(model, optimizer, scheduler, step, output_dir)

        step += 1
        if step >= max_steps:
            break
        pbar.update(1)

    pbar.close()

    # Final save
    save_checkpoint(model, optimizer, scheduler, step, output_dir, is_final=True)
    logger.info("Training completed!")


if __name__ == "__main__":
    fire.Fire(train)
