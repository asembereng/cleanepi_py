#!/usr/bin/env python3
"""
Comprehensive example demonstrating all cleanepi-python features.

This example shows advanced usage including:
- Custom configuration
- Detailed reporting
- Error handling
- Performance monitoring
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import time

# Add the src directory to the path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cleanepi import clean_data, CleaningConfig
from cleanepi.core.config import (
    MissingValueConfig, DuplicateConfig, ConstantConfig, DateConfig
)
from cleanepi.cleaning.standardize_columns import standardize_column_names, suggest_column_renames
from cleanepi.cleaning.replace_missing import (
    replace_missing_values, detect_missing_patterns, suggest_na_strings
)


def create_complex_messy_data():
    """Create a more complex dataset with various data quality issues."""
    np.random.seed(42)
    
    # Create base data
    n_rows = 1000
    data = {
        # ID columns with various formats
        'PATIENT_ID': [f"P{i:04d}" for i in range(1, n_rows + 1)],
        'Study ID': [f"STU-{i:03d}" if i % 10 != 0 else f"STU{i:03d}" for i in range(1, n_rows + 1)],
        
        # Demographics with missing values
        'First Name': np.random.choice(['John', 'Jane', 'Bob', 'Alice', '-99', 'unknown', ''], n_rows),
        'Date_of_Birth': np.random.choice([
            '1990-01-01', '15/02/1985', 'Feb 20, 1992', '1988-12-25', 
            '-99', 'N/A', 'unknown', '1995/03/10'
        ], n_rows),
        'patient age': np.random.choice(['25', '30', 'thirty-five', '40', '-99', 'missing'], n_rows),
        
        # Clinical data
        'Test Result': np.random.choice(['positive', 'negative', 'indeterminate', '1', '0', ''], n_rows),
        'Viral Load': np.random.choice(['1000', '< 50', 'undetectable', 'error', '-', '999999'], n_rows),
        
        # Constant and near-constant columns
        'Laboratory': ['Central Lab'] * n_rows,  # Completely constant
        'Country': ['USA'] * (n_rows - 5) + ['US'] * 5,  # Near-constant
        'Version': ['v1.0'] * n_rows,  # Constant
        
        # Columns with systematic missing patterns
        'Optional Field 1': [''] * n_rows,  # Always empty
        'Optional Field 2': [None] * n_rows,  # Always None
        'Comments': np.random.choice(['Good sample', 'Poor quality', '', 'N/A', '.'], n_rows),
        
        # Mixed data types
        'Mixed Column': np.random.choice([1, 'two', 3.0, '4', None, 'five'], n_rows),
        
        # Duplicate-prone data
        'Sample Type': np.random.choice(['Blood', 'Saliva', 'Urine'] * 10, n_rows),
    }
    
    # Add some exact duplicate rows
    df = pd.DataFrame(data)
    
    # Create duplicates by repeating some rows
    duplicate_indices = np.random.choice(df.index, size=50, replace=False)
    duplicate_rows = df.loc[duplicate_indices].copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def analyze_data_quality(data: pd.DataFrame):
    """Analyze data quality issues in the dataset."""
    print("=" * 60)
    print("DATA QUALITY ANALYSIS")
    print("=" * 60)
    
    print(f"Dataset shape: {data.shape}")
    print(f"Total cells: {data.size:,}")
    
    # Missing value analysis
    missing_counts = data.isna().sum()
    missing_pct = (missing_counts / len(data)) * 100
    
    print(f"\nMissing Values by Column:")
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  {col}: {count} ({missing_pct[col]:.1f}%)")
    
    # Data type analysis
    print(f"\nData Types:")
    for col, dtype in data.dtypes.items():
        print(f"  {col}: {dtype}")
    
    # Detect potential missing patterns
    patterns = detect_missing_patterns(data, threshold=0.02)
    if patterns:
        print(f"\nDetected Missing Patterns:")
        for col, pattern_info in patterns.items():
            print(f"  {col}:")
            for pattern in pattern_info['potential_missing']:
                print(f"    '{pattern['value']}': {pattern['count']} occurrences ({pattern['frequency']:.1%})")
    
    # Suggest column renames
    suggestions = suggest_column_renames(data)
    if suggestions:
        print(f"\nColumn Rename Suggestions:")
        for old, new in suggestions.items():
            print(f"  '{old}' -> '{new}'")
    
    # Duplicate analysis
    total_duplicates = data.duplicated().sum()
    print(f"\nDuplicate Rows: {total_duplicates}")
    
    # Constant column analysis
    constant_cols = []
    for col in data.columns:
        unique_count = data[col].nunique(dropna=False)
        if unique_count <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"\nConstant Columns: {constant_cols}")


def create_comprehensive_config():
    """Create a comprehensive cleaning configuration."""
    return CleaningConfig(
        standardize_column_names=True,
        
        replace_missing_values=MissingValueConfig(
            na_strings=[
                # Standard missing indicators
                '-99', '-999', '99', '999',
                'N/A', 'NA', 'n/a', 'na', 'NULL', 'null',
                'missing', 'unknown', 'error', 'MISSING', 'UNKNOWN',
                '', ' ', '.', '...',
                
                # Clinical-specific
                'undetectable', 'below detection', '< 50',
                'pending', 'not done', 'cancelled',
                
                # Common data entry errors
                '?', '-', '--', '???', 'TBD'
            ],
            custom_na_by_column={
                'viral_load': ['undetectable', '< 50', 'below detection'],
                'test_result': ['error', 'pending', 'cancelled'],
                'patient_age': ['unknown age', 'age not recorded']
            }
        ),
        
        remove_duplicates=DuplicateConfig(
            keep='first'
        ),
        
        remove_constants=ConstantConfig(
            cutoff=0.99,  # Remove columns where 99%+ of values are the same
            exclude_columns=['patient_id']  # Don't remove ID columns even if unique
        ),
        
        # TODO: Add when implemented
        # standardize_dates=DateConfig(
        #     error_tolerance=0.3,
        #     timeframe=('1900-01-01', '2030-12-31')
        # ),
        
        verbose=True,
        strict_validation=False
    )


def benchmark_cleaning(data: pd.DataFrame, config: CleaningConfig):
    """Benchmark the cleaning process."""
    print("=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Memory usage before
    memory_before = data.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage before cleaning: {memory_before:.2f} MB")
    
    # Time the cleaning process
    start_time = time.time()
    cleaned_data, report = clean_data(data, config)
    end_time = time.time()
    
    # Memory usage after
    memory_after = cleaned_data.memory_usage(deep=True).sum() / 1024**2
    memory_saved = memory_before - memory_after
    
    print(f"Cleaning time: {end_time - start_time:.3f} seconds")
    print(f"Memory usage after cleaning: {memory_after:.2f} MB")
    print(f"Memory saved: {memory_saved:.2f} MB ({memory_saved/memory_before*100:.1f}%)")
    print(f"Processing rate: {len(data) / (end_time - start_time):.0f} rows/second")
    
    return cleaned_data, report


def demonstrate_error_handling():
    """Demonstrate error handling capabilities."""
    print("=" * 60)
    print("ERROR HANDLING DEMONSTRATION")
    print("=" * 60)
    
    # Test with empty DataFrame
    try:
        empty_df = pd.DataFrame()
        clean_data(empty_df)
    except ValueError as e:
        print(f"✓ Empty DataFrame error handled: {e}")
    
    # Test with invalid configuration
    try:
        invalid_config = CleaningConfig(
            remove_constants=ConstantConfig(cutoff=1.5)  # Invalid cutoff
        )
    except Exception as e:
        print(f"✓ Invalid configuration error handled: {type(e).__name__}")
    
    # Test with non-DataFrame input
    try:
        from cleanepi.utils.validation import validate_dataframe
        validate_dataframe("not a dataframe")
    except TypeError as e:
        print(f"✓ Type validation error handled: {e}")
    
    print("✓ All error handling tests passed")


def main():
    """Main demonstration function."""
    print("cleanepi-python Comprehensive Example")
    print("=" * 60)
    
    # Create complex messy data
    print("\n1. Creating complex messy dataset...")
    messy_data = create_complex_messy_data()
    print(f"Created dataset with {messy_data.shape[0]} rows and {messy_data.shape[1]} columns")
    
    # Analyze data quality
    print("\n2. Analyzing data quality...")
    analyze_data_quality(messy_data)
    
    # Create comprehensive configuration
    print("\n3. Creating cleaning configuration...")
    config = create_comprehensive_config()
    print("Configuration created with:")
    print(f"  - Column name standardization: {config.standardize_column_names}")
    print(f"  - Missing value patterns: {len(config.replace_missing_values.na_strings)}")
    print(f"  - Duplicate removal: {config.remove_duplicates.keep}")
    print(f"  - Constant column threshold: {config.remove_constants.cutoff}")
    
    # Benchmark cleaning process
    print("\n4. Performing data cleaning with benchmarking...")
    cleaned_data, report = benchmark_cleaning(messy_data, config)
    
    # Show detailed results
    print("\n5. Detailed cleaning results...")
    print("=" * 60)
    
    print(f"Original shape: {messy_data.shape}")
    print(f"Cleaned shape:  {cleaned_data.shape}")
    print(f"Rows removed:   {report.total_rows_removed} ({report.total_rows_removed/len(messy_data)*100:.1f}%)")
    print(f"Columns removed: {report.total_columns_removed}")
    
    # Show column changes
    original_cols = set(messy_data.columns)
    cleaned_cols = set(cleaned_data.columns)
    removed_cols = original_cols - cleaned_cols
    
    if removed_cols:
        print(f"\nRemoved columns: {list(removed_cols)}")
    
    # Show missing value summary
    print(f"\nMissing values after cleaning:")
    missing_summary = cleaned_data.isna().sum()
    missing_summary = missing_summary[missing_summary > 0]
    for col, count in missing_summary.items():
        pct = count / len(cleaned_data) * 100
        print(f"  {col}: {count} ({pct:.1f}%)")
    
    # Show sample of cleaned data
    print(f"\nSample of cleaned data (first 5 rows):")
    print(cleaned_data.head().to_string())
    
    # Export results
    print("\n6. Exporting results...")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Save cleaned data
    cleaned_data.to_csv(output_dir / "cleaned_data.csv", index=False)
    print(f"✓ Cleaned data saved to {output_dir / 'cleaned_data.csv'}")
    
    # Save report
    report.to_file(output_dir / "cleaning_report.json")
    print(f"✓ Cleaning report saved to {output_dir / 'cleaning_report.json'}")
    
    # Save configuration
    config_dict = config.dict()
    import json
    with open(output_dir / "cleaning_config.json", "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"✓ Configuration saved to {output_dir / 'cleaning_config.json'}")
    
    # Demonstrate error handling
    print("\n7. Testing error handling...")
    demonstrate_error_handling()
    
    print("\n" + "=" * 60)
    print("COMPREHENSIVE EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Check the 'output' directory for saved results.")
    print(f"Total processing time: {report.duration:.3f} seconds")
    print(f"Data quality improvement:")
    print(f"  - Removed {report.total_rows_removed} duplicate/invalid rows")
    print(f"  - Removed {report.total_columns_removed} constant/empty columns")
    print(f"  - Standardized {len(cleaned_data.columns)} column names")
    print(f"  - Processed {len(config.replace_missing_values.na_strings)} missing value patterns")


if __name__ == "__main__":
    main()