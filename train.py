#!/usr/bin/env python
"""train.py: Train a Megalodon model using distributed training infrastructure."""

import logging
from pathlib import Path
from typing import Optional

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
from megalodon.utils import clip_grad_norm_, log_host, set_random_seed, setup_env, check_ampere_gpu

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
    Train a Megalodon model using distributed infrastructure.
    Launch with: torchrun --nproc_per_node={NUM_GPUS} train.py [args]

    Args:
        data_dir: Directory with training data
        tokenizer_path: Path to SentencePiece tokenizer
        model_name: Name of model configuration
        output_dir: Directory to save checkpoints
        batch_size: Batch size per GPU
        grad_acc_steps: Gradient accumulation steps
        max_seq_length: Maximum sequence length
        max_steps: Maximum training steps
        save_steps: Save checkpoint every N steps
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

    # Initialize tokenizer and data loader
    tokenizer = Tokenizer(tokenizer_cfg)
    model_cfg.vocab_size = model_cfg.output_size = tokenizer.sp_model_vocab_size
    train_loader = DataLoader(
        tokenizer=tokenizer,
        path=str(Path(data_dir) / "train.jsonl"),
        batch_size=batch_size,
        world_rank=global_rank,
        world_size=world_size,
        chunk_size=model_cfg.chunk_size,
        max_seq_length=max_seq_length,
    )

    # Build model using repo's distributed-aware builder
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
        for batch in train_loader:

            # logger.debug(f"shape of batch['x']: {batch['x'].shape}")
            # logger.debug(f"shape of batch['y']: {batch['y'].shape}")

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

                # if step > 0 and step % log_every_n_steps == 0:
                avg_loss = accumulated_loss / grad_acc_steps
                logger.info(
                    f"Step {step} | Loss: {avg_loss:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )
                accumulated_loss = 0


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
