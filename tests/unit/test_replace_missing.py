"""
Comprehensive tests for the replace_missing module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from cleanepi.cleaning.replace_missing import replace_missing_values


class TestReplaceMissingValues:
    """Test replace_missing_values functionality."""
    
    def test_basic_replacement(self):
        """Test basic missing value replacement."""
        df = pd.DataFrame({
            'col1': ['1', '-99', '3', 'N/A'],
            'col2': ['a', 'NULL', 'c', '']
        })
        
        result = replace_missing_values(df)
        
        # Check that missing values were replaced
        assert result['col1'].isna().sum() >= 1  # At least one NA
        assert result['col2'].isna().sum() >= 1  # At least one NA
        assert not result['col1'].isna().iloc[0]  # First value should not be NA
    
    def test_custom_na_strings(self):
        """Test custom NA strings."""
        df = pd.DataFrame({
            'col1': ['1', 'MISSING', '3', 'UNKNOWN'],
            'col2': ['a', 'b', 'c', 'd']
        })
        
        result = replace_missing_values(df, na_strings=['MISSING', 'UNKNOWN'])
        
        # Check that custom NA strings were replaced
        assert result['col1'].isna().sum() == 2
        assert not result['col2'].isna().any()  # No NAs in col2
    
    def test_target_columns(self):
        """Test processing specific columns."""
        df = pd.DataFrame({
            'process_me': ['1', '-99', '3'],
            'keep_me': ['a', '-99', 'c']
        })
        
        result = replace_missing_values(df, target_columns=['process_me'])
        
        # Only process_me should have NAs
        assert result['process_me'].isna().any()
        assert not result['keep_me'].isna().any()
    
    def test_custom_na_by_column(self):
        """Test column-specific NA strings."""
        df = pd.DataFrame({
            'col1': ['1', 'MISSING1', '3'],
            'col2': ['a', 'MISSING2', 'c']
        })
        
        custom_na = {
            'col1': ['MISSING1'],
            'col2': ['MISSING2']
        }
        
        result = replace_missing_values(df, custom_na_by_column=custom_na)
        
        # Each column should have exactly one NA
        assert result['col1'].isna().sum() == 1
        assert result['col2'].isna().sum() == 1
        assert result['col1'].isna().iloc[1]  # MISSING1 replaced
        assert result['col2'].isna().iloc[1]  # MISSING2 replaced
    
    def test_case_sensitive(self):
        """Test case sensitivity."""
        df = pd.DataFrame({
            'col1': ['1', 'na', '3', 'N/A'],
            'col2': ['a', 'null', 'c', 'NULL']
        })
        
        # Case insensitive (default)
        result_insensitive = replace_missing_values(df, case_sensitive=False)
        
        # Case sensitive
        result_sensitive = replace_missing_values(df, case_sensitive=True)
        
        # Case insensitive should find more matches
        assert result_insensitive['col1'].isna().sum() >= result_sensitive['col1'].isna().sum()
        assert result_insensitive['col2'].isna().sum() >= result_sensitive['col2'].isna().sum()
    
    def test_strip_whitespace(self):
        """Test whitespace stripping."""
        df = pd.DataFrame({
            'col1': ['1', ' -99 ', '3', 'N/A '],
            'col2': ['a', ' NULL', 'c', ' ']
        })
        
        # With whitespace stripping (default)
        result_strip = replace_missing_values(df, strip_whitespace=True)
        
        # Without whitespace stripping
        result_no_strip = replace_missing_values(df, strip_whitespace=False)
        
        # Stripping should find more matches
        assert result_strip['col1'].isna().sum() >= result_no_strip['col1'].isna().sum()
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        
        with pytest.raises(ValueError):  # Should raise validation error
            replace_missing_values(df)
    
    def test_no_missing_values(self):
        """Test with DataFrame that has no missing values to replace."""
        df = pd.DataFrame({
            'col1': ['1', '2', '3'],
            'col2': ['a', 'b', 'c']
        })
        
        result = replace_missing_values(df)
        
        # Should return unchanged DataFrame
        assert not result['col1'].isna().any()
        assert not result['col2'].isna().any()
        pd.testing.assert_frame_equal(result, df)
    
    def test_all_missing_values(self):
        """Test with DataFrame where all values are missing."""
        df = pd.DataFrame({
            'col1': ['-99', 'N/A', 'NULL'],
            'col2': ['', 'missing', 'unknown']
        })
        
        result = replace_missing_values(df)
        
        # Most or all values should be NA
        assert result['col1'].isna().sum() >= 2
        assert result['col2'].isna().sum() >= 2
    
    def test_mixed_types(self):
        """Test with mixed data types."""
        df = pd.DataFrame({
            'numeric': [1, -99, 3.0, np.nan],
            'string': ['a', 'N/A', 'c', ''],
            'boolean': [True, False, True, False]
        })
        
        result = replace_missing_values(df)
        
        # Should handle different types appropriately
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        assert list(result.columns) == list(df.columns)
    
    def test_already_na_values(self):
        """Test with DataFrame that already contains NA values."""
        df = pd.DataFrame({
            'col1': ['1', np.nan, '3', '-99'],
            'col2': ['a', 'b', np.nan, 'NULL']
        })
        
        result = replace_missing_values(df)
        
        # Should preserve existing NAs and add new ones
        assert result['col1'].isna().sum() >= 2  # Original NA + -99
        assert result['col2'].isna().sum() >= 2  # Original NA + NULL
    
    def test_nonexistent_target_columns(self):
        """Test with target columns that don't exist."""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        
        with pytest.raises(ValueError):
            replace_missing_values(df, target_columns=['nonexistent'])
    
    def test_empty_na_strings(self):
        """Test with empty NA strings list."""
        df = pd.DataFrame({
            'col1': ['1', '-99', '3'],
            'col2': ['a', 'NULL', 'c']
        })
        
        result = replace_missing_values(df, na_strings=[])
        
        # Should not replace any values
        assert not result['col1'].isna().any()
        assert not result['col2'].isna().any()
    
    def test_none_values_in_data(self):
        """Test with None values in data."""
        df = pd.DataFrame({
            'col1': ['1', None, '3'],
            'col2': ['a', 'b', None]
        })
        
        result = replace_missing_values(df)
        
        # None values should already be treated as NA
        assert result['col1'].isna().any()
        assert result['col2'].isna().any()
    
    def test_numeric_na_values(self):
        """Test with numeric representations of missing values."""
        df = pd.DataFrame({
            'col1': [1, -99, 3, -999],
            'col2': [1.0, -99.0, 3.0, np.inf]
        })
        
        # Convert to string first to handle numeric NA codes
        df_str = df.astype(str)
        result = replace_missing_values(df_str, na_strings=['-99', '-999'])
        
        assert result['col1'].isna().sum() >= 1
    
    def test_large_dataframe(self):
        """Test with large DataFrame."""
        # Create a larger test DataFrame
        n_rows = 10000
        df = pd.DataFrame({
            'col1': ['value'] * (n_rows - 100) + ['-99'] * 50 + ['N/A'] * 50,
            'col2': ['data'] * (n_rows - 100) + ['NULL'] * 100
        })
        
        result = replace_missing_values(df)
        
        # Should handle large data efficiently
        assert len(result) == n_rows
        assert result['col1'].isna().sum() == 100
        assert result['col2'].isna().sum() == 100
    
    @patch('cleanepi.cleaning.replace_missing.logger')
    def test_logging(self, mock_logger):
        """Test that logging occurs."""
        df = pd.DataFrame({'col1': ['1', '-99', '3']})
        
        replace_missing_values(df)
        
        # Should log information about the operation
        mock_logger.info.assert_called()


@pytest.fixture
def sample_messy_dataframe():
    """Create a sample DataFrame with various missing value representations."""
    return pd.DataFrame({
        'id': ['001', '002', '003', '004', '005'],
        'age': ['25', '-99', '30', 'N/A', ''],
        'status': ['positive', 'unknown', 'negative', '', 'NULL'],
        'score': [85.5, -999, 92.0, np.nan, 78.5],
        'notes': ['good', 'missing', 'excellent', '   ', 'fair']
    })


@pytest.fixture
def custom_na_config():
    """Create custom NA configuration for testing."""
    return {
        'global_na': ['-99', 'N/A', 'NULL', '', 'unknown', 'missing'],
        'column_specific': {
            'score': ['-999'],
            'notes': ['   ']  # whitespace-only string
        }
    }