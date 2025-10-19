#!/usr/bin/env python
"""inference.py: Run inference using a trained Megalodon model"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import fire
import torch
from tqdm.auto import tqdm

from megalodon.logger import initialize_logger, add_logger_file_handler
from megalodon.reloading import reload_model
from megalodon.utils import (
    setup_env,
    log_host,
    set_random_seed,
    check_ampere_gpu,
)
from megalodon.distributed import (
    init_signal_handler,
    init_torch_distributed,
    initialize_model_parallel,
)

logger = logging.getLogger()


def log(t, eps=1e-20):
    """Helper function for gumbel sampling."""
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    """Generate Gumbel noise for sampling."""
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(logits, temperature=1.3, dim=-1):
    """Sample from logits using Gumbel-softmax trick."""
    gumbel = gumbel_noise(logits)
    return ((logits / max(temperature, 1e-10)) + gumbel).argmax(dim=dim)


def min_p_filter(logits, min_p=0.1):
    """Filter logits using minimum probability threshold relative to maximum probability."""
    probs = logits.softmax(dim=-1)
    max_probs = probs.amax(dim=-1, keepdim=True)
    limit = min_p * max_probs
    filtered_logits = torch.where(probs < limit, float("-inf"), logits)
    return filtered_logits


@torch.no_grad()
def generate_completion(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.3,
    min_p: float = 0.1,
    seed: Optional[int] = None,
) -> str:
    """
    Generate text completion for a single prompt.

    Args:
        model: The Megalodon model
        tokenizer: The tokenizer
        prompt: Input prompt text
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        min_p: Minimum probability threshold relative to max probability
        seed: Random seed for reproducibility

    Returns:
        Generated text completion
    """
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()

    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt, bos=True, eos=False)
    x = torch.tensor(prompt_tokens, dtype=torch.long, device="cuda").unsqueeze(0)

    # Generate tokens
    for _ in tqdm(range(max_new_tokens), desc="Generating", leave=False):
        # Get logits for next token
        logits, _ = model(x)
        logits = logits[:, -1, :]  # Last token logits

        # Apply min_p filtering and sample
        logits = min_p_filter(logits, min_p=min_p)
        next_token = gumbel_sample(logits, temperature=temperature).unsqueeze(1)

        # Append and check for EOS
        x = torch.cat([x, next_token], dim=1)
        if next_token.item() == tokenizer.eos_id:
            break

    # Decode full sequence
    generated_tokens = x[0].tolist()
    text = tokenizer.decode(generated_tokens, cut_at_eos=True)

    return text


def process_file(
    model,
    tokenizer,
    input_file: Path,
    output_file: Path,
    max_new_tokens: int = 100,
    temperature: float = 1.3,
    min_p: float = 0.1,
    seed: Optional[int] = None,
) -> None:
    """
    Process a file containing prompts and generate completions.

    Args:
        model: The Megalodon model
        tokenizer: The tokenizer
        input_file: Path to input file (one prompt per line)
        output_file: Path to save generations
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        min_p: Minimum probability threshold
        seed: Random seed
    """
    prompts = input_file.read_text().strip().split("\n")

    with open(output_file, "w", encoding="utf-8") as f:
        for prompt in tqdm(prompts, desc="Processing prompts"):
            completion = generate_completion(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                min_p=min_p,
                seed=seed,
            )
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Completion: {completion}\n")
            f.write("-" * 80 + "\n")


def run_inference(
    checkpoint_dir: str,
    output_dir: Optional[str] = None,
    input_file: Optional[str] = None,
    prompt: Optional[str] = None,
    max_new_tokens: int = 100,
    temperature: float = 1.3,
    min_p: float = 0.1,
    model_parallel_size: int = 1,
    chunk_parallel_size: int = 1,
    seed: int = 42,
    dtype: str = "fp32",
    distributed_timeout: int = 1800,
) -> None:
    """
    Run inference using a trained Megalodon model.

    Args:
        checkpoint_dir: Directory containing model checkpoint
        model_name: Model architecture name (used if config.json not found)
        output_dir: Directory to save outputs (default: checkpoint_dir/generations)
        input_file: Optional path to file containing prompts (one per line)
        prompt: Optional single prompt to process
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (higher = more random)
        min_p: Minimum probability threshold relative to max probability
        model_parallel_size: Model parallel size
        chunk_parallel_size: Chunk parallel size
        seed: Random seed
        dtype: Model dtype for inference
        distributed_timeout: Distributed timeout in seconds
    """


    # Basic setup
    initialize_logger()
    setup_env()
    log_host()
    set_random_seed(seed)
    check_ampere_gpu()

    checkpoint_dir = Path(checkpoint_dir).resolve()  # Get absolute path

    output_dir = Path(output_dir) if output_dir else checkpoint_dir / "generations"
    output_dir.mkdir(parents=True, exist_ok=True)
    add_logger_file_handler(output_dir / "inference.log")

    # Initialize distributed setup
    init_signal_handler()
    logger.info("Initializing distributed training...")
    _, global_rank, world_size = init_torch_distributed(timeout=distributed_timeout)

    logger.info("Initializing model parallel...")
    initialize_model_parallel(model_parallel_size, chunk_parallel_size)

    # Load model - use actual API from reload_model
    logger.info(f"Loading model from {checkpoint_dir}...")
    model, tokenizer, _ = reload_model(
        checkpoint_dir=str(checkpoint_dir),
        init_distributed=False,  # We already initialized
        model_parallel_size=model_parallel_size,
        chunk_parallel_size=chunk_parallel_size,
        dtype=dtype,
        tokenizer_path=str(checkpoint_dir / "tokenizer.model"),
    )

    # Handle input modes
    if input_file and prompt:
        raise ValueError("Specify either input_file or prompt, not both")

    if input_file:
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        output_path = output_dir / f"generations_{input_path.stem}.txt"
        logger.info(f"Processing prompts from {input_file}")
        process_file(
            model,
            tokenizer,
            input_path,
            output_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            min_p=min_p,
            seed=seed,
        )
        logger.info(f"Saved generations to {output_path}")

    elif prompt:
        logger.info("Generating from single prompt...")
        completion = generate_completion(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            min_p=min_p,
            seed=seed,
        )
        logger.info("\nGenerated completion:")
        logger.info("-" * 80)
        logger.info(completion)
        logger.info("-" * 80)

        # Also print to console for interactive use
        print("\nGenerated completion:")
        print("-" * 80)
        print(completion)
        print("-" * 80)

    else:
        raise ValueError("Must specify either input_file or prompt")


if __name__ == "__main__":
    fire.Fire(run_inference)