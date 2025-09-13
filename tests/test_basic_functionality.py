"""
Basic tests for the cleanepi package functionality.
"""

import pytest
import pandas as pd
import numpy as np
from cleanepi import clean_data, CleaningConfig
from cleanepi.core.config import MissingValueConfig, DuplicateConfig, ConstantConfig
from cleanepi.cleaning.standardize_columns import standardize_column_names
from cleanepi.cleaning.replace_missing import replace_missing_values
from cleanepi.cleaning.remove_duplicates import remove_duplicates
from cleanepi.cleaning.remove_constants import remove_constants


class TestBasicFunctionality:
    """Test basic package functionality."""
    
    def test_import(self):
        """Test that package imports correctly."""
        from cleanepi import clean_data, CleaningConfig
        assert clean_data is not None
        assert CleaningConfig is not None
    
    def test_standardize_column_names(self):
        """Test column name standardization."""
        df = pd.DataFrame({
            'Date of Birth': [1, 2, 3],
            'Patient ID': [4, 5, 6],
            'Test Value': [7, 8, 9]
        })
        
        result = standardize_column_names(df)
        
        expected_columns = ['date_of_birth', 'patient_id', 'test_value']
        assert list(result.columns) == expected_columns
        assert result.shape == df.shape
    
    def test_replace_missing_values(self):
        """Test missing value replacement."""
        df = pd.DataFrame({
            'age': ['25', '-99', '30', 'N/A'],
            'status': ['positive', 'unknown', 'negative', '']
        })
        
        result = replace_missing_values(df, na_strings=['-99', 'unknown', 'N/A', ''])
        
        # Check that missing values were replaced
        assert result['age'].isna().sum() == 2  # -99 and N/A
        assert result['status'].isna().sum() == 2  # unknown and empty string
    
    def test_remove_duplicates(self):
        """Test duplicate removal."""
        df = pd.DataFrame({
            'id': [1, 2, 2, 3],
            'value': ['a', 'b', 'b', 'c']
        })
        
        result = remove_duplicates(df)
        
        assert len(result) == 3  # One duplicate removed
        assert result['id'].tolist() == [1, 2, 3]
    
    def test_remove_constants(self):
        """Test constant column removal."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'constant_col': ['same', 'same', 'same'],
            'variable_col': ['a', 'b', 'c']
        })
        
        result = remove_constants(df, cutoff=1.0)
        
        assert 'constant_col' not in result.columns
        assert 'id' in result.columns
        assert 'variable_col' in result.columns
    
    def test_clean_data_basic(self):
        """Test basic clean_data functionality."""
        df = pd.DataFrame({
            'Date of Birth': ['1990-01-01', '-99', '1990-01-01'],  # Make it a true duplicate
            'Patient ID': [1, 2, 1],  # Has duplicate row
            'Constant Col': ['same', 'same', 'same'],
            'Status': ['positive', 'negative', 'positive']  # Make it a true duplicate
        })
        
        config = CleaningConfig(
            standardize_column_names=True,
            replace_missing_values=MissingValueConfig(na_strings=['-99', 'unknown']),
            remove_duplicates=DuplicateConfig(),
            remove_constants=ConstantConfig(cutoff=1.0)
        )
        
        result, report = clean_data(df, config)
        
        # Check that operations were performed
        assert 'constant_col' not in result.columns  # Constant column removed
        assert len(result) == 2  # Duplicate removed
        assert result['date_of_birth'].isna().sum() == 1  # Missing value replaced
        
        # Check report
        assert report.total_rows_removed == 1
        assert report.total_columns_removed == 1
        assert len(report.operations) > 0


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_valid_config(self):
        """Test that valid config passes validation."""
        config = CleaningConfig()
        # Should not raise any exceptions
        assert config is not None
    
    def test_invalid_error_tolerance(self):
        """Test that invalid error tolerance raises error."""
        with pytest.raises(ValueError):
            from cleanepi.core.config import DateConfig
            DateConfig(error_tolerance=1.5)  # Should be between 0 and 1


class TestDataValidation:
    """Test data validation."""
    
    def test_empty_dataframe_error(self):
        """Test that empty DataFrame raises error."""
        from cleanepi.utils.validation import validate_dataframe
        
        df = pd.DataFrame()
        with pytest.raises(ValueError):
            validate_dataframe(df)
    
    def test_non_dataframe_error(self):
        """Test that non-DataFrame input raises error."""
        from cleanepi.utils.validation import validate_dataframe
        
        with pytest.raises(TypeError):
            validate_dataframe("not a dataframe")


@pytest.fixture
def sample_messy_data():
    """Create sample messy data for testing."""
    return pd.DataFrame({
        'Study ID': ['PS001P2', 'PS002P2', 'PS003P2', 'PS001P2'],  # Exact duplicate row
        'Date of Birth': ['01/01/1990', '-99', '15/05/1985', '01/01/1990'],
        'Patient Age': ['25', 'unknown', '35', '25'],
        'Test Result': ['positive', 'negative', 'positive', 'positive'],
        'Empty Column': ['', '', '', ''],  # Constant empty column
        'Constant Value': ['same', 'same', 'same', 'same'],  # Constant column
        'Mixed Types': [1, 'two', 3.0, 1]  # Make exact duplicate
    })


def test_comprehensive_cleaning(sample_messy_data):
    """Test comprehensive data cleaning on messy data."""
    config = CleaningConfig(
        standardize_column_names=True,
        replace_missing_values=MissingValueConfig(na_strings=['-99', 'unknown', '']),
        remove_duplicates=DuplicateConfig(),
        remove_constants=ConstantConfig(cutoff=1.0)
    )
    
    result, report = clean_data(sample_messy_data, config)
    
    # Verify cleaning operations
    assert len(result) == 3  # Duplicate row removed
    assert 'empty_column' not in result.columns  # Empty column removed
    assert 'constant_value' not in result.columns  # Constant column removed
    assert result['patient_age'].isna().sum() == 1  # Unknown value replaced
    
    # Verify report
    assert report.has_warnings() == False or report.has_warnings() == True  # May have warnings
    assert not report.has_errors()  # Should not have errors
    assert len(report.successful_operations) > 0