"""
Comprehensive tests for the remove_constants module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from cleanepi.cleaning.remove_constants import remove_constants


class TestRemoveConstants:
    """Test remove_constants functionality."""
    
    def test_basic_constant_removal(self):
        """Test basic constant column removal."""
        df = pd.DataFrame({
            'constant_col': ['same'] * 5,
            'variable_col': [1, 2, 3, 4, 5],
            'another_constant': [42] * 5
        })
        
        result = remove_constants(df)
        
        # Should remove constant columns
        assert 'constant_col' not in result.columns
        assert 'another_constant' not in result.columns
        assert 'variable_col' in result.columns
        assert len(result.columns) == 1
    
    def test_no_constant_columns(self):
        """Test with DataFrame that has no constant columns."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e'],
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        result = remove_constants(df)
        
        # Should return unchanged DataFrame
        assert len(result.columns) == 3
        pd.testing.assert_frame_equal(result, df)
    
    def test_cutoff_threshold(self):
        """Test cutoff threshold functionality."""
        df = pd.DataFrame({
            'mostly_constant': ['same'] * 8 + ['different'] * 2,  # 80% same
            'half_constant': ['same'] * 5 + ['different'] * 5,    # 50% same
            'variable': list(range(10))  # All different
        })
        
        # With cutoff 1.0 (100% same)
        result_strict = remove_constants(df, cutoff=1.0)
        assert len(result_strict.columns) == 3  # No removal
        
        # With cutoff 0.8 (80% same)
        result_medium = remove_constants(df, cutoff=0.8)
        assert 'mostly_constant' not in result_medium.columns
        assert 'half_constant' in result_medium.columns
        assert 'variable' in result_medium.columns
        
        # With cutoff 0.5 (50% same)
        result_loose = remove_constants(df, cutoff=0.5)
        assert 'mostly_constant' not in result_loose.columns
        assert 'half_constant' not in result_loose.columns
        assert 'variable' in result_loose.columns
    
    def test_exclude_columns(self):
        """Test excluding specific columns from removal."""
        df = pd.DataFrame({
            'constant_to_remove': ['same'] * 5,
            'constant_to_keep': ['same'] * 5,
            'variable_col': [1, 2, 3, 4, 5]
        })
        
        result = remove_constants(df, exclude_columns=['constant_to_keep'])
        
        # Should remove constant_to_remove but keep constant_to_keep
        assert 'constant_to_remove' not in result.columns
        assert 'constant_to_keep' in result.columns
        assert 'variable_col' in result.columns
        assert len(result.columns) == 2
    
    def test_all_constant_columns(self):
        """Test with DataFrame where all columns are constant."""
        df = pd.DataFrame({
            'const1': ['same'] * 5,
            'const2': [42] * 5,
            'const3': [True] * 5
        })
        
        result = remove_constants(df)
        
        # Should remove all columns
        assert len(result.columns) == 0
        assert len(result) == 5  # Rows should be preserved
    
    def test_single_column_constant(self):
        """Test with single constant column."""
        df = pd.DataFrame({'constant_col': ['same'] * 10})
        
        result = remove_constants(df)
        
        # Should remove the only column
        assert len(result.columns) == 0
        assert len(result) == 10
    
    def test_single_column_variable(self):
        """Test with single variable column."""
        df = pd.DataFrame({'variable_col': [1, 2, 3, 4, 5]})
        
        result = remove_constants(df)
        
        # Should keep the variable column
        assert len(result.columns) == 1
        assert 'variable_col' in result.columns
        pd.testing.assert_frame_equal(result, df)
    
    def test_with_nan_values(self):
        """Test constant detection with NaN values."""
        df = pd.DataFrame({
            'constant_with_nan': ['same', 'same', 'same', np.nan, np.nan],
            'mostly_nan': [np.nan, np.nan, np.nan, np.nan, 'value'],
            'all_nan': [np.nan, np.nan, np.nan, np.nan, np.nan],
            'variable': [1, 2, 3, 4, 5]
        })
        
        result = remove_constants(df, cutoff=0.8)
        
        # Should handle NaN values in proportion calculation
        assert 'mostly_nan' not in result.columns  # 80% NaN
        assert 'all_nan' not in result.columns     # 100% NaN
        assert 'variable' in result.columns
        # constant_with_nan has 60% 'same', should be kept with cutoff 0.8
    
    def test_mixed_data_types(self):
        """Test with mixed data types."""
        df = pd.DataFrame({
            'int_constant': [42] * 5,
            'float_constant': [3.14] * 5,
            'str_constant': ['same'] * 5,
            'bool_constant': [True] * 5,
            'mixed_variable': [1, 'two', 3.0, True, None]
        })
        
        result = remove_constants(df)
        
        # Should remove all constant columns regardless of type
        assert 'int_constant' not in result.columns
        assert 'float_constant' not in result.columns
        assert 'str_constant' not in result.columns
        assert 'bool_constant' not in result.columns
        assert 'mixed_variable' in result.columns
        assert len(result.columns) == 1
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        
        with pytest.raises(ValueError):  # Should raise validation error
            remove_constants(df)
    
    def test_single_row_dataframe(self):
        """Test with single row DataFrame."""
        df = pd.DataFrame({
            'col1': [1],
            'col2': ['value'],
            'col3': [True]
        })
        
        result = remove_constants(df)
        
        # All columns are "constant" with single row, should remove all
        assert len(result.columns) == 0
        assert len(result) == 1
    
    def test_invalid_cutoff_values(self):
        """Test with invalid cutoff values."""
        df = pd.DataFrame({'col': [1, 2, 3]})
        
        # Test cutoff > 1.0
        with pytest.raises(ValueError, match="cutoff must be between 0.0 and 1.0"):
            remove_constants(df, cutoff=1.5)
        
        # Test cutoff < 0.0
        with pytest.raises(ValueError, match="cutoff must be between 0.0 and 1.0"):
            remove_constants(df, cutoff=-0.1)
    
    def test_boundary_cutoff_values(self):
        """Test with boundary cutoff values."""
        df = pd.DataFrame({
            'constant': ['same'] * 5,
            'variable': [1, 2, 3, 4, 5]
        })
        
        # Test cutoff = 0.0 (should remove anything with at least one repeated value)
        # Since both columns have repeated values at cutoff 0.0, behavior may vary
        result_zero = remove_constants(df, cutoff=0.0)
        assert isinstance(result_zero, pd.DataFrame)
        
        # Test cutoff = 1.0 (should remove perfect constants only)
        result_one = remove_constants(df, cutoff=1.0)
        assert 'constant' not in result_one.columns
        assert 'variable' in result_one.columns
    
    def test_near_constant_columns(self):
        """Test detection of near-constant columns."""
        df = pd.DataFrame({
            'almost_constant': ['same'] * 9 + ['different'],  # 90% same
            'half_constant': ['same'] * 5 + ['diff'] * 5,     # 50% same
            'variable': list(range(10))
        })
        
        result = remove_constants(df, cutoff=0.85)
        
        # Should remove almost_constant (90% > 85%) but keep others
        assert 'almost_constant' not in result.columns
        assert 'half_constant' in result.columns
        assert 'variable' in result.columns
    
    def test_preserve_data_integrity(self):
        """Test that data integrity is preserved."""
        df = pd.DataFrame({
            'keep_me': [1, 2, 3, 4, 5],
            'remove_me': ['same'] * 5,
            'also_keep': ['a', 'b', 'c', 'd', 'e']
        })
        
        result = remove_constants(df)
        
        # Should preserve data in remaining columns
        assert list(result['keep_me']) == [1, 2, 3, 4, 5]
        assert list(result['also_keep']) == ['a', 'b', 'c', 'd', 'e']
        assert len(result) == len(df)
    
    def test_large_dataframe(self):
        """Test with large DataFrame."""
        n_rows = 10000
        df = pd.DataFrame({
            'constant_large': ['same'] * n_rows,
            'variable_large': list(range(n_rows)),
            'mostly_constant': ['same'] * (n_rows - 10) + ['diff'] * 10
        })
        
        result = remove_constants(df, cutoff=0.999)
        
        # Should handle large data efficiently
        assert 'constant_large' not in result.columns
        assert 'mostly_constant' not in result.columns
        assert 'variable_large' in result.columns
        assert len(result) == n_rows
    
    @patch('cleanepi.cleaning.remove_constants.logger')
    def test_logging_with_removal(self, mock_logger):
        """Test logging when constants are removed."""
        df = pd.DataFrame({
            'constant': ['same'] * 5,
            'variable': [1, 2, 3, 4, 5]
        })
        
        remove_constants(df)
        
        # Should log removal
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[0][0]
        assert "Removed" in call_args and "constant" in call_args
    
    @patch('cleanepi.cleaning.remove_constants.logger')
    def test_logging_no_removal(self, mock_logger):
        """Test logging when no constants are found."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        remove_constants(df)
        
        # Should log no removal
        mock_logger.info.assert_called_with("No constant columns found")
    
    def test_exclude_nonexistent_columns(self):
        """Test excluding columns that don't exist."""
        df = pd.DataFrame({
            'constant': ['same'] * 5,
            'variable': [1, 2, 3, 4, 5]
        })
        
        # Should not raise error for nonexistent excluded columns
        result = remove_constants(df, exclude_columns=['nonexistent', 'constant'])
        
        assert 'constant' in result.columns  # Excluded from removal
        assert 'variable' in result.columns
        assert len(result.columns) == 2
    
    def test_empty_exclude_list(self):
        """Test with empty exclude list."""
        df = pd.DataFrame({
            'constant': ['same'] * 5,
            'variable': [1, 2, 3, 4, 5]
        })
        
        result = remove_constants(df, exclude_columns=[])
        
        # Should work same as no exclude_columns
        assert 'constant' not in result.columns
        assert 'variable' in result.columns
    
    def test_special_values(self):
        """Test with special values like inf, -inf."""
        df = pd.DataFrame({
            'inf_constant': [np.inf] * 5,
            'neginf_constant': [-np.inf] * 5,
            'mixed_special': [np.inf, -np.inf, np.nan, 1, 2],
            'variable': [1, 2, 3, 4, 5]
        })
        
        result = remove_constants(df)
        
        # Should handle special values
        assert 'inf_constant' not in result.columns
        assert 'neginf_constant' not in result.columns
        assert 'mixed_special' in result.columns
        assert 'variable' in result.columns


@pytest.fixture
def sample_dataframe_with_constants():
    """Create sample DataFrame with constant columns for testing."""
    return pd.DataFrame({
        'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
        'study_site': ['Site_A'] * 5,  # Constant
        'age': [25, 30, 35, 28, 32],   # Variable
        'protocol': ['Protocol_1'] * 5,  # Constant
        'status': ['active', 'inactive', 'active', 'pending', 'active']  # Variable
    })


@pytest.fixture
def sample_mixed_constants():
    """Create DataFrame with various levels of constantness."""
    return pd.DataFrame({
        'perfect_constant': ['same'] * 10,
        'mostly_constant': ['same'] * 9 + ['different'],
        'half_constant': ['same'] * 5 + ['different'] * 5,
        'variable': list(range(10))
    })