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


def load_config(config_path: str) -> CleaningConfig:
    """Load configuration from JSON file."""
    with open(config_path) as f:
        config_dict = json.load(f)
    return CleaningConfig(**config_dict)


def process_streaming(args) -> tuple:
    """Process data using streaming mode."""
    try:
        from cleanepi.performance.streaming import StreamingCleaner
    except ImportError:
        raise ImportError("Streaming processing requires performance extras to be installed")
    
    logger.info(f"Using streaming processing with chunk size: {args.chunk_size}")
    
    config = load_config(args.config) if args.config else create_config_from_args(args)
    
    streaming_cleaner = StreamingCleaner(
        chunk_size=args.chunk_size,
        memory_limit=args.memory_limit
    )
    
    def progress_callback(chunk_num, total_chunks, message):
        if total_chunks > 0:
            logger.info(f"Progress: {chunk_num}/{total_chunks} - {message}")
        else:
            logger.info(f"Progress: {chunk_num} chunks - {message}")
    
    cleaned_data, report = streaming_cleaner.clean_csv_streaming(
        args.input,
        config,
        output_path=args.output,
        progress_callback=progress_callback
    )
    
    return cleaned_data, report


def process_async(args) -> tuple:
    """Process data using async mode."""
    import asyncio
    
    try:
        from cleanepi.performance.async_processing import AsyncCleaner
    except ImportError:
        raise ImportError("Async processing requires async extras to be installed")
    
    logger.info(f"Using async processing with {args.max_workers} workers")
    
    async def run_async_processing():
        config = load_config(args.config) if args.config else create_config_from_args(args)
        
        async def progress_callback(step, total_steps, message):
            logger.info(f"Progress: {step}/{total_steps} - {message}")
        
        async with AsyncCleaner(max_workers=args.max_workers) as async_cleaner:
            cleaned_data, report = await async_cleaner.clean_csv(
                args.input,
                config,
                output_path=args.output,
                progress_callback=progress_callback
            )
        
        return cleaned_data, report
    
    return asyncio.run(run_async_processing())


def process_dask(args) -> tuple:
    """Process data using Dask mode."""
    try:
        from cleanepi.performance.dask_processing import DaskCleaner
    except ImportError:
        raise ImportError("Dask processing requires performance extras to be installed")
    
    logger.info("Using Dask processing")
    
    config = load_config(args.config) if args.config else create_config_from_args(args)
    
    with DaskCleaner(memory_limit=args.memory_limit) as dask_cleaner:
        cleaned_data, report = dask_cleaner.clean_csv(
            args.input,
            config,
            output_path=args.output,
            chunk_size=args.memory_limit
        )
    
    return cleaned_data, report


def process_distributed(args) -> tuple:
    """Process data using distributed mode."""
    try:
        from cleanepi.performance.distributed import DistributedCleaner
    except ImportError:
        raise ImportError("Distributed processing requires performance extras to be installed")
    
    logger.info(f"Using distributed processing")
    if args.scheduler_address:
        logger.info(f"Connecting to scheduler: {args.scheduler_address}")
    
    config = load_config(args.config) if args.config else create_config_from_args(args)
    
    with DistributedCleaner(scheduler_address=args.scheduler_address) as dist_cleaner:
        cleaned_data, report = dist_cleaner.process_large_csv_distributed(
            args.input,
            config,
            output_path=args.output,
            chunk_size=args.chunk_size
        )
    
    return cleaned_data, report
    """Create cleaning configuration from command line arguments."""
    
    # Missing value config
    missing_config = None
    if args.na_strings or args.replace_missing:
        na_strings = args.na_strings.split(',') if args.na_strings else None
        missing_config = MissingValueConfig(na_strings=na_strings)
    
    # Duplicate config
    duplicate_config = None
    if args.remove_duplicates:
        duplicate_config = DuplicateConfig(keep=args.duplicate_keep)
    
    # Constant config
    constant_config = None
    if args.remove_constants:
        constant_config = ConstantConfig(cutoff=args.constant_cutoff)
    
    # Date standardization config
    date_config = None
    if args.standardize_dates:
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
    if args.validate_subject_ids:
        target_columns = args.subject_id_columns.split(',') if args.subject_id_columns else []
        subject_id_config = SubjectIDConfig(
            target_columns=target_columns,
            prefix=args.subject_id_prefix,
            suffix=args.subject_id_suffix,
            nchar=args.subject_id_length
        )
    
    # Numeric conversion config
    numeric_config = None
    if args.convert_numeric:
        target_columns = args.numeric_columns.split(',') if args.numeric_columns else []
        numeric_config = NumericConfig(
            target_columns=target_columns,
            lang=args.numeric_language,
            errors=args.numeric_errors
        )
    
    # Dictionary cleaning
    dictionary = None
    if args.dictionary_file:
        with open(args.dictionary_file) as f:
            dictionary = json.load(f)
    
    # Date sequence validation
    date_sequence_columns = None
    if args.check_date_sequence:
        date_sequence_columns = args.date_sequence_columns.split(',') if args.date_sequence_columns else None
    
    return CleaningConfig(
        standardize_column_names=args.standardize_columns,
        replace_missing_values=missing_config,
        remove_duplicates=duplicate_config,
        remove_constants=constant_config,
        standardize_dates=date_config,
        standardize_subject_ids=subject_id_config,
        to_numeric=numeric_config,
        dictionary=dictionary,
        check_date_sequence=date_sequence_columns,
        verbose=args.verbose
    )


def create_config_from_args(args) -> CleaningConfig:
    """Create CleaningConfig from command line arguments."""
    # Missing value configuration
    missing_config = None
    if args.replace_missing:
        na_strings = None
        if args.na_strings:
            na_strings = [s.strip() for s in args.na_strings.split(',')]
        missing_config = MissingValueConfig(na_strings=na_strings)
    
    # Duplicate configuration
    duplicate_config = None
    if args.remove_duplicates:
        keep = args.duplicate_keep if args.duplicate_keep != "False" else False
        duplicate_config = DuplicateConfig(keep=keep)
    
    # Constant configuration
    constant_config = None
    if args.remove_constants:
        constant_config = ConstantConfig(cutoff=args.constant_cutoff)
    
    # Date configuration
    date_config = None
    if args.standardize_dates:
        target_columns = None
        if args.date_columns:
            target_columns = [s.strip() for s in args.date_columns.split(',')]
        
        timeframe = None
        if args.date_timeframe:
            dates = args.date_timeframe.split(',')
            if len(dates) == 2:
                timeframe = (dates[0].strip(), dates[1].strip())
        
        date_config = DateConfig(
            target_columns=target_columns,
            timeframe=timeframe,
            error_tolerance=args.date_error_tolerance
        )
    
    # Subject ID configuration
    subject_id_config = None
    if args.validate_subject_ids:
        target_columns = None
        if args.subject_id_columns:
            target_columns = [s.strip() for s in args.subject_id_columns.split(',')]
        
        subject_id_config = SubjectIDConfig(
            target_columns=target_columns,
            prefix=args.subject_id_prefix,
            suffix=args.subject_id_suffix,
            character_length=args.subject_id_length
        )
    
    # Numeric configuration
    numeric_config = None
    if args.convert_numeric:
        target_columns = None
        if args.numeric_columns:
            target_columns = [s.strip() for s in args.numeric_columns.split(',')]
        
        numeric_config = NumericConfig(
            target_columns=target_columns,
            language=args.numeric_language,
            errors=args.numeric_errors
        )
    
    # Dictionary cleaning
    dictionary_file = args.dictionary_file if hasattr(args, 'dictionary_file') else None
    
    # Date sequence checking
    date_sequence_columns = None
    if hasattr(args, 'check_date_sequence') and args.check_date_sequence:
        if args.date_sequence_columns:
            date_sequence_columns = [s.strip() for s in args.date_sequence_columns.split(',')]
    
    return CleaningConfig(
        standardize_column_names=args.standardize_columns,
        replace_missing_values=missing_config,
        remove_duplicates=duplicate_config,
        remove_constants=constant_config,
        standardize_dates=date_config,
        validate_subject_ids=subject_id_config,
        convert_numeric=numeric_config,
        dictionary_file=dictionary_file,
        check_date_sequence=date_sequence_columns is not None,
        date_sequence_columns=date_sequence_columns
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
    
    # Performance and scalability options
    parser.add_argument(
        "--processing-mode",
        choices=["standard", "streaming", "async", "dask", "distributed"],
        default="standard",
        help="Processing mode for large datasets"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Chunk size for streaming/distributed processing"
    )
    parser.add_argument(
        "--memory-limit",
        default="500MB",
        help="Memory limit for processing (e.g., '500MB', '2GB')"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of worker threads/processes"
    )
    parser.add_argument(
        "--scheduler-address",
        help="Dask scheduler address for distributed processing"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark instead of cleaning data"
    )
    parser.add_argument(
        "--benchmark-scalability",
        action="store_true",
        help="Run scalability benchmark with different data sizes"
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
        # Handle benchmark modes
        if args.benchmark or args.benchmark_scalability:
            from cleanepi.performance.benchmarking import PerformanceBenchmark, run_comprehensive_benchmark
            
            if args.benchmark_scalability:
                logger.info("Running comprehensive scalability benchmark")
                results_path = run_comprehensive_benchmark(
                    output_dir=Path(args.output).parent if args.output else None,
                    max_rows=100000
                )
                print(f"Benchmark results saved to: {results_path}")
                return
            
            elif args.benchmark:
                logger.info("Running performance benchmark")
                data = load_data(args.input)
                config = create_config_from_args(args) if not args.config else load_config(args.config)
                
                benchmark = PerformanceBenchmark(
                    output_dir=Path(args.output).parent if args.output else None
                )
                
                # Run benchmark
                result = benchmark.benchmark_operation(data, config, "cli_benchmark")
                
                # Save and print results
                results_path = benchmark.save_results()
                report = benchmark.generate_report()
                
                print("\nBenchmark Results:")
                print(report)
                print(f"\nDetailed results saved to: {results_path}")
                return
        
        logger.info(f"Loading data from {args.input}")
        
        # Handle different processing modes
        if args.processing_mode == "streaming":
            cleaned_data, report = process_streaming(args)
        elif args.processing_mode == "async":
            cleaned_data, report = process_async(args)
        elif args.processing_mode == "dask":
            cleaned_data, report = process_dask(args)
        elif args.processing_mode == "distributed":
            cleaned_data, report = process_distributed(args)
        else:
            # Standard processing
            data = load_data(args.input)
            logger.info(f"Loaded data with shape {data.shape}")
            
            # Create configuration
            if args.config:
                config = load_config(args.config)
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