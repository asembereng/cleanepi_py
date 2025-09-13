#!/usr/bin/env python3
"""
Command-line interface for cleanepi.

Provides a simple CLI for data cleaning operations.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from cleanepi import CleaningConfig, clean_data
from cleanepi.core.config import ConstantConfig, DuplicateConfig, MissingValueConfig
from cleanepi.utils.validation import detect_encoding, validate_file_safety


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level, format="{time} | {level} | {message}")


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from file."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Validate file safety
    validate_file_safety(str(path))

    # Load based on extension
    if path.suffix.lower() == ".csv":
        encoding = detect_encoding(str(path))
        return pd.read_csv(path, encoding=encoding)
    elif path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    elif path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def save_data(data: pd.DataFrame, file_path: str):
    """Save data to file."""
    path = Path(file_path)

    # Create directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save based on extension
    if path.suffix.lower() == ".csv":
        data.to_csv(path, index=False)
    elif path.suffix.lower() in [".xlsx", ".xls"]:
        data.to_excel(path, index=False)
    elif path.suffix.lower() == ".parquet":
        data.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")


def create_config_from_args(args) -> CleaningConfig:
    """Create cleaning configuration from command line arguments."""

    # Missing value config
    missing_config = None
    if args.na_strings or args.replace_missing:
        na_strings = args.na_strings.split(",") if args.na_strings else None
        missing_config = MissingValueConfig(na_strings=na_strings)

    # Duplicate config
    duplicate_config = None
    if args.remove_duplicates:
        duplicate_config = DuplicateConfig(keep=args.duplicate_keep)

    # Constant config
    constant_config = None
    if args.remove_constants:
        constant_config = ConstantConfig(cutoff=args.constant_cutoff)

    return CleaningConfig(
        standardize_column_names=args.standardize_columns,
        replace_missing_values=missing_config,
        remove_duplicates=duplicate_config,
        remove_constants=constant_config,
        verbose=args.verbose,
    )


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Clean and standardize epidemiological data", prog="cleanepi"
    )

    # Input/Output
    parser.add_argument("input", help="Input data file (CSV, Excel, Parquet)")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--config", help="JSON configuration file")

    # Cleaning options
    parser.add_argument(
        "--standardize-columns",
        action="store_true",
        help="Standardize column names to snake_case",
    )
    parser.add_argument(
        "--replace-missing", action="store_true", help="Replace missing values with NA"
    )
    parser.add_argument(
        "--na-strings",
        help="Comma-separated list of strings to treat as missing values",
    )
    parser.add_argument(
        "--remove-duplicates", action="store_true", help="Remove duplicate rows"
    )
    parser.add_argument(
        "--duplicate-keep",
        choices=["first", "last", "False"],
        default="first",
        help="Which duplicates to keep",
    )
    parser.add_argument(
        "--remove-constants", action="store_true", help="Remove constant columns"
    )
    parser.add_argument(
        "--constant-cutoff",
        type=float,
        default=1.0,
        help="Proportion threshold for constant columns (0.0-1.0)",
    )

    # Output options
    parser.add_argument("--report", help="Path to save cleaning report (JSON)")
    parser.add_argument(
        "--preview", type=int, default=0, help="Number of rows to preview in output"
    )

    # General options
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument("--version", action="version", version="cleanepi 0.1.0")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    try:
        logger.info(f"Loading data from {args.input}")
        data = load_data(args.input)
        logger.info(f"Loaded data with shape {data.shape}")

        # Create configuration
        if args.config:
            with open(args.config) as f:
                config_dict = json.load(f)
            config = CleaningConfig(**config_dict)
        else:
            config = create_config_from_args(args)

        # Clean data
        logger.info("Starting data cleaning")
        cleaned_data, report = clean_data(data, config)

        # Save output
        if args.output:
            logger.info(f"Saving cleaned data to {args.output}")
            save_data(cleaned_data, args.output)
        else:
            # Default output name
            input_path = Path(args.input)
            output_path = (
                input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"
            )
            logger.info(f"Saving cleaned data to {output_path}")
            save_data(cleaned_data, str(output_path))

        # Save report
        if args.report:
            logger.info(f"Saving report to {args.report}")
            report.to_file(args.report)

        # Print summary
        print("\nCleaning Summary:")
        print(f"  Original shape: {data.shape}")
        print(f"  Cleaned shape:  {cleaned_data.shape}")
        print(f"  Rows removed:   {report.total_rows_removed}")
        print(f"  Columns removed: {report.total_columns_removed}")

        # Preview data
        if args.preview > 0:
            print(f"\nFirst {args.preview} rows of cleaned data:")
            print(cleaned_data.head(args.preview).to_string())

        logger.info("Data cleaning completed successfully")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
