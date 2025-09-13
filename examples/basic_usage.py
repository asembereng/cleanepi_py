#!/usr/bin/env python3
"""
Example usage of the cleanepi-python package.

This script demonstrates basic data cleaning functionality.
"""

import pandas as pd
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cleanepi import clean_data, CleaningConfig
from cleanepi.core.config import MissingValueConfig, DuplicateConfig, ConstantConfig


def create_sample_data():
    """Create sample messy epidemiological data."""
    data = {
        'Study ID': ['PS001P2', 'PS002P2', 'PS003P2', 'PS001P2', 'PS004P2'],
        'Date of Birth': ['01/01/1990', '-99', '15/05/1985', '01/01/1990', 'N/A'],
        'Patient Age': ['25', 'unknown', '35', '25', '42'],
        'Test Result': ['positive', 'negative', 'positive', 'positive', 'negative'],
        'Status Code': ['1', '2', '1', '1', '99'],
        'Empty Column': ['', '', '', '', ''],
        'Constant Value': ['same', 'same', 'same', 'same', 'same'],
        'Country Code': ['2', '2', '2', '2', '2'],
        'mixed CASE columns': ['a', 'b', 'c', 'a', 'e']
    }
    
    return pd.DataFrame(data)


def main():
    """Main example function."""
    print("cleanepi-python Example")
    print("=" * 40)
    
    # Create sample data
    print("\n1. Creating sample messy data...")
    df = create_sample_data()
    print(f"Original data shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Configure cleaning operations
    print("\n2. Configuring cleaning operations...")
    config = CleaningConfig(
        standardize_column_names=True,
        replace_missing_values=MissingValueConfig(
            na_strings=['-99', 'unknown', 'N/A', '']
        ),
        remove_duplicates=DuplicateConfig(
            keep='first'
        ),
        remove_constants=ConstantConfig(
            cutoff=1.0  # Remove columns where 100% of values are the same
        )
    )
    
    # Perform cleaning
    print("\n3. Performing data cleaning...")
    try:
        cleaned_df, report = clean_data(df, config)
        
        print(f"Cleaned data shape: {cleaned_df.shape}")
        print(f"Cleaned columns: {list(cleaned_df.columns)}")
        
        print("\nCleaned data:")
        print(cleaned_df)
        
        # Print cleaning report
        print("\n4. Cleaning Report:")
        print("-" * 40)
        report.print_summary()
        
        # Show specific changes
        print("\n5. Specific Changes:")
        print(f"  - Rows removed: {report.total_rows_removed}")
        print(f"  - Columns removed: {report.total_columns_removed}")
        print(f"  - Missing values: {cleaned_df.isna().sum().sum()}")
        
        # Show missing value patterns
        print("\n6. Missing Value Summary by Column:")
        missing_summary = cleaned_df.isna().sum()
        missing_summary = missing_summary[missing_summary > 0]
        if len(missing_summary) > 0:
            for col, count in missing_summary.items():
                percentage = (count / len(cleaned_df)) * 100
                print(f"  - {col}: {count} missing ({percentage:.1f}%)")
        else:
            print("  No missing values found")
        
        return cleaned_df, report
        
    except Exception as e:
        print(f"Error during cleaning: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    cleaned_data, cleaning_report = main()
    
    if cleaned_data is not None:
        print("\nExample completed successfully!")
    else:
        print("\nExample failed!")
        sys.exit(1)