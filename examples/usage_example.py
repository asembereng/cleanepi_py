"""Example usage of the cleanepi package."""

# This script shows how to use the cleanepi package once dependencies are installed

import pandas as pd
import numpy as np
from datetime import date

# Import cleanepi functions
from cleanepi import (
    clean_data,
    replace_missing_values,
    remove_constants,
    find_and_remove_duplicates,
    standardize_date,
    convert_to_numeric,
    print_report
)

def create_sample_data():
    """Create sample epidemiological data for testing."""
    return pd.DataFrame({
        'study_id': ['PS001P2', 'PS002P2', 'PS004P2-1', 'PS003P2', 'P0005P2', 'PS006P2', 'PS002P2'],  # duplicate
        'event_name': ['day 0', 'day 0', 'day 0', 'day 0', 'day 0', 'day 0', 'day 0'],
        'country_code': [2, 2, 2, 2, 2, 2, 2],  # constant column
        'country_name': ['Gambia', 'Gambia', 'Gambia', 'Gambia', 'Gambia', 'Gambia', 'Gambia'],  # constant
        'date_of_admission': ['01/12/2020', '28/01/2021', '15/02/2021', '11/02/2021', '17/02/2021', '17/02/2021', '28/01/2021'],
        'date_of_birth': ['06/01/1972', '02/20/1952', '06/15/1961', '11/11/1947', '09/26/2000', '-99', '02/20/1952'],
        'date_first_pcr_positive_test': ['Dec 01, 2020', 'Jan 01, 2021', 'Feb 11, 2021', 'Feb 01, 2021', 'Feb 16, 2021', 'May 02, 2021', 'Jan 01, 2021'],
        'sex': [1, 1, -99, 1, 2, 2, 1],
        'age': ['25', 'thirty', '35', '40', 'twenty-one', 'missing', 'thirty'],
        'empty_column': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]  # completely empty
    })


def example_individual_functions():
    """Example of using individual cleaning functions."""
    print("=== EXAMPLE: Using Individual Functions ===\n")
    
    data = create_sample_data()
    print("Original data shape:", data.shape)
    print("Original data:")
    print(data.to_string())
    print()
    
    # 1. Replace missing values
    print("1. Replacing missing values...")
    data = replace_missing_values(data, na_strings=['-99', 'missing'])
    print_report(data, 'missing_values_replaced_at')
    print()
    
    # 2. Remove constant columns and empty columns
    print("2. Removing constant columns...")
    data = remove_constants(data, cutoff=1.0)
    print_report(data, 'constant_data')
    print()
    
    # 3. Remove duplicates
    print("3. Removing duplicates...")
    data = find_and_remove_duplicates(data)
    print_report(data, 'removed_duplicates')
    print()
    
    # 4. Standardize dates
    print("4. Standardizing dates...")
    timeframe = [date(1950, 1, 1), date(2025, 12, 31)]
    data = standardize_date(data, timeframe=timeframe)
    print_report(data, 'standardized_dates')
    print()
    
    # 5. Convert to numeric
    print("5. Converting to numeric...")
    data = convert_to_numeric(data, target_columns=['sex', 'age'], lang='en')
    print_report(data, 'converted_into_numeric')
    print()
    
    print("Final cleaned data shape:", data.shape)
    print("Final cleaned data:")
    print(data.to_string())
    print()


def example_clean_data_function():
    """Example of using the main clean_data function."""
    print("=== EXAMPLE: Using Main clean_data Function ===\n")
    
    data = create_sample_data()
    print("Original data shape:", data.shape)
    
    # Define cleaning parameters
    cleaning_params = {
        'replace_missing_values': {
            'target_columns': None,
            'na_strings': ['-99', 'missing']
        },
        'remove_duplicates': {
            'target_columns': None
        },
        'remove_constants': {
            'cutoff': 1.0
        },
        'standardize_dates': {
            'target_columns': ['date_of_admission', 'date_of_birth', 'date_first_pcr_positive_test'],
            'timeframe': [date(1950, 1, 1), date(2025, 12, 31)],
            'error_tolerance': 0.4
        },
        'standardize_subject_ids': {
            'target_columns': ['study_id'],
            'prefix': 'PS',
            'suffix': 'P2',
            'range': (1, 100),
            'nchar': 7
        },
        'to_numeric': {
            'target_columns': ['sex', 'age'],
            'lang': 'en'
        }
    }
    
    # Perform all cleaning operations at once
    cleaned_data = clean_data(data, **cleaning_params)
    
    print("Cleaned data shape:", cleaned_data.shape)
    print("Cleaned data:")
    print(cleaned_data.to_string())
    print()
    
    # Print comprehensive report
    print("=== COMPREHENSIVE CLEANING REPORT ===")
    print_report(cleaned_data)


def example_subject_id_checking():
    """Example of subject ID format checking."""
    print("=== EXAMPLE: Subject ID Checking ===\n")
    
    # Create data with various ID format issues
    data = pd.DataFrame({
        'study_id': ['PS001P2', 'PS002P2', 'XY003P2', '', 'PS004P2', 'PS002P2', 'PS123P2'],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Bob', 'Frank']
    })
    
    print("Data with ID issues:")
    print(data.to_string())
    print()
    
    from cleanepi import standardize_subject_ids
    
    # Check subject ID format
    result = standardize_subject_ids(
        data,
        target_columns=['study_id'],
        prefix='PS',
        suffix='P2', 
        range=(1, 100),
        nchar=7
    )
    
    print_report(result, 'subject_id_check')


if __name__ == "__main__":
    print("CLEANEPI PYTHON PACKAGE - USAGE EXAMPLES")
    print("=" * 50)
    print()
    
    print("NOTE: This example requires pandas, numpy, and python-dateutil to be installed.")
    print("Install with: pip install pandas numpy python-dateutil")
    print()
    
    try:
        # Run examples
        example_individual_functions()
        print("\n" + "=" * 50 + "\n")
        
        example_clean_data_function()
        print("\n" + "=" * 50 + "\n")
        
        example_subject_id_checking()
        
        print("\n🎉 All examples completed successfully!")
        
    except ImportError as e:
        print(f"❌ Missing required dependency: {e}")
        print("Please install required packages: pip install pandas numpy python-dateutil")
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()