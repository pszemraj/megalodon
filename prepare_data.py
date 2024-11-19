"""
Script to convert HuggingFace datasets to JSONL format with configurable text column.

Example usage:
    python prepare_data.py \
        --dataset_name "pszemraj/simple_wikipedia_LM" \
        --text_column "text"
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any

import fire
from datasets import DatasetDict, load_dataset
from logging import getLogger, StreamHandler, INFO
from tqdm import tqdm

logger = getLogger(__name__)
logger.setLevel(INFO)
logger.addHandler(StreamHandler())


def setup_output_dir(output_dir: Optional[Path] = None, dataset_name: str = "") -> Path:
    """
    Setup and create output directory if it doesn't exist.

    Args:
        output_dir: Optional custom output directory
        dataset_name: Name of dataset to use in default path

    Returns:
        Path: Configured output directory
    """
    if output_dir is None:
        # Create default output path based on dataset name

        safe_name = dataset_name.replace("/", "_")
        output_dir = Path("data") / safe_name
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_dataset_to_jsonl(
    dataset_dict: DatasetDict,
    output_dir: Path,
    text_column: str = "text",
    additional_columns: Optional[list] = None,
) -> Dict[str, Path]:
    """
    Save each split of the dataset to a separate JSONL file.

    Args:
        dataset_dict: The loaded dataset dictionary
        output_dir: Directory to save the JSONL files
        text_column: Name of the column containing text data
        additional_columns: List of additional columns to include besides text_column

    Returns:
        Dict mapping split names to output file paths
    """
    output_files = {}

    for split in dataset_dict.keys():
        split_dataset = dataset_dict[split]
        output_path = output_dir / f"{split}.jsonl"
        output_files[split] = output_path

        logger.info(f"Processing {split} split -> {output_path}")

        with output_path.open("w", encoding="utf-8") as f:
            for example in tqdm(split_dataset, desc=f"Writing {split}"):
                # Start with text column

                if text_column not in example:
                    raise KeyError(
                        f"Text column '{text_column}' not found in dataset. "
                        f"Available columns: {list(example.keys())}"
                    )
                line = {"text": example[text_column]}

                # Add any additional requested columns

                if additional_columns:
                    for col in additional_columns:
                        if col in example and col != text_column:
                            line[col] = example[col]
                f.write(json.dumps(line) + "\n")
    return output_files


def convert_to_jsonl(
    dataset_name: str,
    dataset_config: Optional[str] = None,
    text_column: str = "text",
    additional_columns: Optional[list] = None,
    output_dir: Optional[str] = None,
    **dataset_kwargs: Any,
) -> Dict[str, Path]:
    """
    Convert a HuggingFace dataset to JSONL format.

    Args:
        dataset_name: Name of dataset on HuggingFace Hub
        dataset_config: Optional dataset configuration name
        text_column: Name of the column containing text data
        additional_columns: Optional list of additional columns to include
        output_dir: Optional custom output directory
        **dataset_kwargs: Additional kwargs passed to load_dataset

    Returns:
        Dict mapping split names to output file paths
    """
    logger.info(f"Loading dataset '{dataset_name}'...")

    try:
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, **dataset_kwargs)
        else:
            dataset = load_dataset(dataset_name, **dataset_kwargs)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
    logger.info("Dataset loaded successfully")

    # Setup output directory

    output_path = setup_output_dir(
        Path(output_dir) if output_dir else None, dataset_name
    )
    logger.info(f"Output directory: {output_path}")

    # Convert and save dataset

    output_files = save_dataset_to_jsonl(
        dataset, output_path, text_column, additional_columns
    )

    logger.info(f"All splits saved to {output_path}")
    return output_files


def main():
    """CLI entrypoint for data preparation."""
    fire.Fire(convert_to_jsonl)


if __name__ == "__main__":
    main()
