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

from cleanepi import clean_data, CleaningConfig
from cleanepi.core.config import (
    MissingValueConfig, DuplicateConfig, ConstantConfig,
    DateConfig, SubjectIDConfig, NumericConfig
)
from cleanepi.utils.validation import validate_file_safety, detect_encoding


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
    if path.suffix.lower() == '.csv':
        encoding = detect_encoding(str(path))
        return pd.read_csv(path, encoding=encoding)
    elif path.suffix.lower() in ['.xlsx', '.xls']:
        return pd.read_excel(path)
    elif path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def save_data(data: pd.DataFrame, file_path: str):
    """Save data to file."""
    path = Path(file_path)
    
    # Create directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save based on extension
    if path.suffix.lower() == '.csv':
        data.to_csv(path, index=False)
    elif path.suffix.lower() in ['.xlsx', '.xls']:
        data.to_excel(path, index=False)
    elif path.suffix.lower() == '.parquet':
        data.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")


def create_config_from_args(args) -> CleaningConfig:
    """Create cleaning configuration from command line arguments."""
    
    # Missing value config
    missing_config = None
    if args.na_strings or (getattr(args, 'replace_missing', False) is True):
        na_strings = args.na_strings.split(',') if args.na_strings else None
        missing_config = MissingValueConfig(na_strings=na_strings)
    
    # Duplicate config
    duplicate_config = None
    if getattr(args, 'remove_duplicates', False) is True:
        duplicate_config = DuplicateConfig(keep=args.duplicate_keep)
    
    # Constant config
    constant_config = None
    if getattr(args, 'remove_constants', False) is True:
        constant_config = ConstantConfig(cutoff=args.constant_cutoff)
    
    # Date standardization config
    date_config = None
    if getattr(args, 'standardize_dates', False) is True:
        target_columns = args.date_columns.split(',') if args.date_columns else None
        timeframe = None
        if args.date_timeframe:
            timeframe_parts = args.date_timeframe.split(',')
            if len(timeframe_parts) == 2:
                timeframe = (timeframe_parts[0].strip(), timeframe_parts[1].strip())
        
        date_config = DateConfig(
            target_columns=target_columns,
            timeframe=timeframe,
            error_tolerance=args.date_error_tolerance
        )
    
    # Subject ID validation config
    subject_id_config = None
    if getattr(args, 'validate_subject_ids', False) is True:
        target_columns = args.subject_id_columns.split(',') if args.subject_id_columns else []
        subject_id_config = SubjectIDConfig(
            target_columns=target_columns,
            prefix=(args.subject_id_prefix if isinstance(args.subject_id_prefix, (str, type(None))) else None),
            suffix=(args.subject_id_suffix if isinstance(args.subject_id_suffix, (str, type(None))) else None),
            nchar=(args.subject_id_length if isinstance(args.subject_id_length, (int, type(None))) else None)
        )
    
    # Numeric conversion config
    numeric_config = None
    if getattr(args, 'convert_numeric', False) is True:
        target_columns = args.numeric_columns.split(',') if args.numeric_columns else []
        numeric_config = NumericConfig(
            target_columns=target_columns,
            lang=args.numeric_language,
            errors=args.numeric_errors
        )
    
    # Dictionary cleaning
    dictionary = None
    dict_path = getattr(args, 'dictionary_file', None)
    if isinstance(dict_path, str) and dict_path.strip():
        with open(dict_path) as f:
            dictionary = json.load(f)
    
    # Date sequence validation
    date_sequence_columns = None
    if getattr(args, 'check_date_sequence', False) is True:
        date_sequence_columns = args.date_sequence_columns.split(',') if args.date_sequence_columns else None
    
    return CleaningConfig(
        standardize_column_names=(getattr(args, 'standardize_columns', False) is True),
        replace_missing_values=missing_config,
        remove_duplicates=duplicate_config,
        remove_constants=constant_config,
        standardize_dates=date_config,
        standardize_subject_ids=subject_id_config,
        to_numeric=numeric_config,
        dictionary=dictionary,
        check_date_sequence=date_sequence_columns,
        verbose=(getattr(args, 'verbose', False) is True)
    )


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Clean and standardize epidemiological data",
        prog="cleanepi"
    )
    
    # Input/Output
    parser.add_argument("input", help="Input data file (CSV, Excel, Parquet)")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--config", help="JSON configuration file")
    
    # Cleaning options
    parser.add_argument(
        "--standardize-columns", 
        action="store_true",
        help="Standardize column names to snake_case"
    )
    parser.add_argument(
        "--replace-missing",
        action="store_true", 
        help="Replace missing values with NA"
    )
    parser.add_argument(
        "--na-strings",
        help="Comma-separated list of strings to treat as missing values"
    )
    parser.add_argument(
        "--remove-duplicates",
        action="store_true",
        help="Remove duplicate rows"
    )
    parser.add_argument(
        "--duplicate-keep",
        choices=["first", "last", "False"],
        default="first",
        help="Which duplicates to keep"
    )
    parser.add_argument(
        "--remove-constants",
        action="store_true",
        help="Remove constant columns"
    )
    parser.add_argument(
        "--constant-cutoff",
        type=float,
        default=1.0,
        help="Proportion threshold for constant columns (0.0-1.0)"
    )
    
    # Advanced cleaning options
    parser.add_argument(
        "--standardize-dates",
        action="store_true",
        help="Standardize date columns with intelligent parsing"
    )
    parser.add_argument(
        "--date-columns",
        help="Comma-separated list of date columns to standardize (auto-detect if not specified)"
    )
    parser.add_argument(
        "--date-timeframe",
        help="Valid date range as 'start_date,end_date' in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--date-error-tolerance",
        type=float,
        default=0.4,
        help="Proportion of unparseable dates to tolerate (0.0-1.0)"
    )
    
    parser.add_argument(
        "--validate-subject-ids",
        action="store_true",
        help="Validate subject ID columns"
    )
    parser.add_argument(
        "--subject-id-columns",
        help="Comma-separated list of subject ID columns to validate"
    )
    parser.add_argument(
        "--subject-id-prefix",
        help="Expected prefix for subject IDs"
    )
    parser.add_argument(
        "--subject-id-suffix", 
        help="Expected suffix for subject IDs"
    )
    parser.add_argument(
        "--subject-id-length",
        type=int,
        help="Expected total character length for subject IDs"
    )
    
    parser.add_argument(
        "--convert-numeric",
        action="store_true",
        help="Convert text columns to numeric with intelligent parsing"
    )
    parser.add_argument(
        "--numeric-columns",
        help="Comma-separated list of columns to convert to numeric"
    )
    parser.add_argument(
        "--numeric-language",
        default="en",
        choices=["en", "es", "fr"],
        help="Language for number word conversion (e.g., 'one' -> 1)"
    )
    parser.add_argument(
        "--numeric-errors",
        default="coerce",
        choices=["raise", "coerce", "ignore"],
        help="How to handle conversion errors"
    )
    
    parser.add_argument(
        "--dictionary-file",
        help="JSON file containing dictionary mappings for value replacement"
    )
    
    parser.add_argument(
        "--check-date-sequence",
        action="store_true",
        help="Check date sequence validity and logical ordering"
    )
    parser.add_argument(
        "--date-sequence-columns",
        help="Comma-separated list of date columns to check in chronological order"
    )
    
    # Output options
    parser.add_argument(
        "--report",
        help="Path to save cleaning report (JSON)"
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Number of rows to preview in output"
    )
    
    # General options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="cleanepi 0.1.0"
    )
    
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
            output_path = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"
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