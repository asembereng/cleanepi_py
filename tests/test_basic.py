"""Basic tests for cleanepi package."""

import pandas as pd
import numpy as np
from datetime import date
import sys
import os

# Add the package to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cleanepi import (
    clean_data,
    replace_missing_values,
    remove_constants,
    find_and_remove_duplicates,
    convert_to_numeric,
    standardize_date,
    print_report,
    get_report
)


def test_replace_missing_values():
    """Test the replace_missing_values function."""
    print("Testing replace_missing_values...")
    
    data = pd.DataFrame({
        'col1': ['A', '-99', 'C', 'missing'],
        'col2': ['1', '2', 'NULL', '4']
    })
    
    result = replace_missing_values(data, na_strings=['-99', 'missing', 'NULL'])
    
    # Check that missing values were replaced
    assert pd.isna(result.loc[1, 'col1'])  # '-99' should be NaN
    assert pd.isna(result.loc[3, 'col1'])  # 'missing' should be NaN  
    assert pd.isna(result.loc[2, 'col2'])  # 'NULL' should be NaN
    
    print("✅ replace_missing_values test passed")


def test_remove_constants():
    """Test the remove_constants function."""
    print("Testing remove_constants...")
    
    data = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'constant_col': [1, 1, 1, 1],
        'empty_col': [np.nan, np.nan, np.nan, np.nan],
        'mixed_col': ['A', 'B', np.nan, 'D']
    })
    
    result = remove_constants(data, cutoff=1.0)
    
    # Constant and empty columns should be removed
    assert 'constant_col' not in result.columns
    assert 'empty_col' not in result.columns
    assert 'id' in result.columns
    assert 'mixed_col' in result.columns
    
    print("✅ remove_constants test passed")


def test_find_and_remove_duplicates():
    """Test the find_and_remove_duplicates function."""
    print("Testing find_and_remove_duplicates...")
    
    data = pd.DataFrame({
        'id': [1, 2, 2, 3],
        'name': ['A', 'B', 'B', 'C'],
        'value': [10, 20, 20, 30]
    })
    
    result = find_and_remove_duplicates(data)
    
    # Should have 3 unique rows
    assert len(result) == 3
    assert len(result[result['id'] == 2]) == 1  # Only one instance of id=2
    
    print("✅ find_and_remove_duplicates test passed")


def test_convert_to_numeric():
    """Test the convert_to_numeric function."""
    print("Testing convert_to_numeric...")
    
    data = pd.DataFrame({
        'age': ['25', 'thirty', '35', '40'],
        'score': ['10', '20', '30', '40']
    })
    
    result = convert_to_numeric(data, target_columns=['score'], lang='en')
    
    # Score column should be converted to numeric
    assert pd.api.types.is_numeric_dtype(result['score'])
    
    print("✅ convert_to_numeric test passed")


def test_standardize_date():
    """Test the standardize_date function.""" 
    print("Testing standardize_date...")
    
    data = pd.DataFrame({
        'date_col': ['2020-01-01', '01/02/2020', 'Jan 3, 2020'],
        'other_col': ['A', 'B', 'C']
    })
    
    timeframe = [date(2019, 1, 1), date(2021, 12, 31)]
    result = standardize_date(data, target_columns=['date_col'], timeframe=timeframe)
    
    # Date column should be converted to datetime
    assert pd.api.types.is_datetime64_any_dtype(result['date_col'])
    
    print("✅ standardize_date test passed")


def test_clean_data():
    """Test the main clean_data function."""
    print("Testing clean_data...")
    
    # Create sample data
    data = pd.DataFrame({
        'study_id': ['PS001P2', 'PS002P2', 'PS003P2', 'PS002P2'],  # duplicate
        'date_column': ['2020-01-01', '28/01/2021', 'Jan 3, 2021', '2021-02-15'],
        'age': ['25', 'thirty', '35', '-99'],  # mixed types, missing value
        'constant_col': [1, 1, 1, 1],  # constant
        'empty_col': [np.nan, np.nan, np.nan, np.nan]  # empty
    })
    
    # Define cleaning parameters
    result = clean_data(
        data,
        replace_missing_values={'target_columns': None, 'na_strings': ['-99']},
        remove_duplicates={'target_columns': None},
        remove_constants={'cutoff': 1.0},
        standardize_dates={
            'target_columns': ['date_column'],
            'timeframe': [date(2020, 1, 1), date(2022, 12, 31)]
        },
        to_numeric={'target_columns': ['age'], 'lang': 'en'}
    )
    
    # Check results
    assert len(result) < len(data)  # Duplicates should be removed
    assert 'constant_col' not in result.columns  # Constant columns removed
    assert 'empty_col' not in result.columns  # Empty columns removed
    assert pd.api.types.is_datetime64_any_dtype(result['date_column'])  # Dates standardized
    
    # Check that report exists
    report = get_report(result)
    assert isinstance(report, dict)
    assert len(report) > 0
    
    print("✅ clean_data test passed")


def test_report_functionality():
    """Test the report functionality."""
    print("Testing report functionality...")
    
    data = pd.DataFrame({
        'col1': ['A', '-99', 'C'],
        'col2': ['1', '2', '3']
    })
    
    result = replace_missing_values(data, na_strings=['-99'])
    
    # Test get_report
    report = get_report(result)
    assert 'missing_values_replaced_at' in report
    assert 'col1' in report['missing_values_replaced_at']
    
    # Test print_report (should not raise errors)
    print_report(result, 'missing_values_replaced_at', print_output=False)
    
    print("✅ report functionality test passed")


if __name__ == "__main__":
    print("Running cleanepi tests...\n")
    
    try:
        test_replace_missing_values()
        test_remove_constants()
        test_find_and_remove_duplicates()
        test_convert_to_numeric()
        test_standardize_date()
        test_clean_data()
        test_report_functionality()
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)