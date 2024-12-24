#!/usr/bin/env python
"""train.py: Train a Megalodon model using distributed training infrastructure"""

import itertools
import logging
import random
from pathlib import Path

import fire
import torch
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
def run_validation(model, val_iter, max_batches: int = 100):
    """
    Run validation, cycling through data as needed.
    Uses an infinite iterator to avoid StopIteration.
    """
    model.eval()
    total_loss = 0
    total_tokens = 0

    for _ in range(max_batches):
        batch = next(val_iter)  # never raises StopIteration
        pred, _ = model(batch["x"].cuda())
        loss = cross_entropy(pred, batch["y"].cuda())
        non_pad_mask = batch["y"].cuda() != -100
        total_loss += loss.sum().item()
        total_tokens += non_pad_mask.sum().item()

    model.train()
    return total_loss / max(total_tokens, 1)


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(logits, temperature=1.3, dim=-1):
    # Add Gumbel noise for sampling
    gumbel = gumbel_noise(logits)
    # Scale by temperature
    return ((logits / max(temperature, 1e-10)) + gumbel).argmax(dim=dim)


def min_p_filter(logits, min_p=0.1):
    # Convert logits to probabilities
    probs = logits.softmax(dim=-1)
    # Get the maximum probability in each row
    max_probs = probs.amax(dim=-1, keepdim=True)
    # Filter out tokens with probability < min_p * max_prob
    limit = min_p * max_probs
    filtered_logits = torch.where(probs < limit, float("-inf"), logits)
    return filtered_logits


@torch.no_grad()
def generate_from_prompt(
    model, tokenizer, prompt_tokens, gen_length=50, temperature=1.3, min_p=0.1
):
    """
    Generate a sequence of tokens using the given model and prompt tokens.
    """
    model.eval()
    x = torch.tensor(prompt_tokens, dtype=torch.long, device="cuda").unsqueeze(0)

    for _ in range(gen_length):
        # Get logits for the current sequence
        logits, _ = model(x)
        logits = logits[:, -1, :]  # Last token logits

        # Apply min_p_filter
        logits = min_p_filter(logits, min_p=min_p)

        # Sample using Gumbel sampling
        next_token = gumbel_sample(logits, temperature=temperature).unsqueeze(1)

        # Append next token
        x = torch.cat([x, next_token], dim=1)

    # Decode the full generated sequence
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
    gen_length: int = 50,
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
    log_every_n_steps: int = 8,
):
    """
    Train a Megalodon model using distributed infrastructure, with validation & generation checks.
    Launch with: torchrun --nproc_per_node={NUM_GPUS} train.py [args]

    NOTE: This version wraps dataloaders in infinite iterators to prevent StopIteration.
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

    # Train dataloader (wrapped with infinite cycle)
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
    # Use itertools.cycle to get an infinite iterator
    train_iter = itertools.cycle(train_loader)

    # Validation dataloader (optional, also wrapped in infinite cycle)
    val_file = Path(data_dir) / "validation.jsonl"
    val_loader = None
    val_iter = None
    if val_file.is_file():
        val_loader_obj = DataLoader(
            tokenizer=tokenizer,
            path=str(val_file),
            batch_size=batch_size,
            world_rank=global_rank,
            world_size=world_size,
            chunk_size=model_cfg.chunk_size,
            max_seq_length=max_seq_length,
        )
        # Infinite iterator for validation data
        val_iter_obj = itertools.cycle(val_loader_obj)

        logger.info(
            "Validation file found. Will run periodic validation and generation checks."
        )
        val_loader = val_loader_obj
        val_iter = val_iter_obj
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
        model=model, cfg=optim_cfg, total_steps=max_steps
    )

    # Training loop
    logger.info("Starting training...")
    model.train()
    step = 0
    optimizer.zero_grad()
    accumulated_loss = 0
    pbar = tqdm(total=max_steps, mininterval=10, desc="Training")

    while step < max_steps:
        # Grab next batch from infinite iterator
        batch = next(train_iter)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            pred, _ = model(batch["x"].cuda())
            # Sum loss across all tokens
            loss = cross_entropy(pred, batch["y"].cuda())
            # Count valid tokens
            non_pad_mask = batch["y"].cuda() != -100
            num_valid = non_pad_mask.sum().item()
            # Per-token loss for logging
            loss_value = loss.sum().item() / max(num_valid, 1)

        # Accumulate
        accumulated_loss += loss_value / grad_acc_steps
        # Scale for gradient accumulation
        loss = loss.sum() / max(num_valid, 1) / grad_acc_steps
        loss.backward()

        # Gradient accumulation boundary
        if (step + 1) % grad_acc_steps == 0:
            # Sync and clip
            model.grad_all_reduce()
            grad_norm = clip_grad_norm_(fsdp_module=model, max_norm=optim_cfg.clip)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Logging
            if (step + 1) % log_every_n_steps == 0:
                logger.info(
                    f"Step {step//grad_acc_steps} | "
                    f"Loss: {accumulated_loss:.4f} | "
                    f"Grad norm: {grad_norm:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )
            accumulated_loss = 0

        # Validation step
        if (
            val_loader is not None
            and val_iter is not None
            and (step > 0)
            and (step % validation_steps == 0)
        ):
            logger.info("Running validation...")
            val_loss = run_validation(model, val_iter, max_batches=100)
            logger.info(
                f"Step {step//grad_acc_steps} | Validation Loss: {val_loss:.4f}"
            )

        # Generation step
        if (
            val_loader is not None
            and val_iter is not None
            and (step > 0)
            and (step % generation_steps == 0)
        ):
            # Grab a batch from val to build a prompt
            batch = next(val_iter)
            x_tokens = batch["x"][0].tolist()
            prompt_length = min(50, len(x_tokens))
            start_idx = random.randint(0, max(0, len(x_tokens) - prompt_length - 1))
            prompt_tokens = x_tokens[start_idx : start_idx + prompt_length]

            prompt_text = tokenizer.decode(prompt_tokens, cut_at_eos=False)
            logger.info(f"\n\n[Generation Step] Prompt:\n{prompt_text}\n{'='*50}")
            generated_text = generate_from_prompt(
                model,
                tokenizer,
                prompt_tokens,
                gen_length=gen_length,
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
    tokenizer.sp_model.save(output_dir / "tokenizer.model")
    logger.info("Training completed!")


if __name__ == "__main__":
    fire.Fire(train)
