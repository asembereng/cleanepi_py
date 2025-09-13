"""Tests for date standardization functionality."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date

from cleanepi.cleaning.standardize_dates import (
    standardize_dates,
    detect_date_columns,
    get_default_date_formats,
    _is_date_like,
    _parse_date_column
)


class TestDetectDateColumns:
    """Test date column detection functionality."""
    
    def test_detect_by_column_names(self):
        """Test detection based on column names."""
        data = pd.DataFrame({
            'patient_id': ['P001', 'P002'],
            'birth_date': ['1990-01-15', '1985-03-20'],
            'visit_time': ['2023-01-10', '2023-02-15'],
            'age': [33, 38],
            'test_result': ['pos', 'neg']
        })
        
        detected = detect_date_columns(data)
        assert 'birth_date' in detected
        assert 'visit_time' in detected
        assert 'patient_id' not in detected
        assert 'age' not in detected
    
    def test_detect_by_content_patterns(self):
        """Test detection based on content patterns."""
        data = pd.DataFrame({
            'col1': ['2023-01-15', '2023-02-20', '2023-03-10'],
            'col2': ['15/01/2023', '20/02/2023', '10/03/2023'],
            'col3': ['text', 'data', 'values'],
            'col4': [1, 2, 3]
        })
        
        detected = detect_date_columns(data)
        assert 'col1' in detected
        assert 'col2' in detected
        assert 'col3' not in detected
        assert 'col4' not in detected


class TestDateLikeness:
    """Test _is_date_like helper function."""
    
    def test_various_date_formats(self):
        """Test recognition of various date formats."""
        date_strings = [
            '2023-01-15',
            '15/01/2023',
            '01-15-2023', 
            '15 Jan 2023',
            'Jan 15, 2023',
            '20230115'
        ]
        
        for date_str in date_strings:
            assert _is_date_like(date_str), f"Should recognize {date_str} as date-like"
    
    def test_non_date_strings(self):
        """Test rejection of non-date strings."""
        non_date_strings = [
            'text',
            '12345',
            'positive',
            '',
            '999-999-9999'  # Invalid date format
        ]
        
        for non_date_str in non_date_strings:
            assert not _is_date_like(non_date_str), f"Should not recognize {non_date_str} as date-like"


class TestStandardizeDates:
    """Test date standardization functionality."""
    
    def test_basic_date_standardization(self):
        """Test basic date standardization with mixed formats."""
        data = pd.DataFrame({
            'birth_date': ['1990-01-15', '15/02/1985', '1995-07-10'],
            'visit_date': ['2023-01-10', '2023/02/15', '2023-01-05']
        })
        
        result = standardize_dates(data, target_columns=['birth_date', 'visit_date'])
        
        # Check that columns are converted to datetime
        assert pd.api.types.is_datetime64_any_dtype(result['birth_date'])
        assert pd.api.types.is_datetime64_any_dtype(result['visit_date'])
        
        # Check that dates are parsed correctly
        assert result['birth_date'].iloc[0] == pd.Timestamp('1990-01-15')
        assert result['birth_date'].iloc[1] == pd.Timestamp('1985-02-15')
    
    def test_auto_detection(self):
        """Test automatic date column detection."""
        data = pd.DataFrame({
            'patient_id': ['P001', 'P002'],
            'birth_date': ['1990-01-15', '1985-03-20'],
            'age': [33, 38]
        })
        
        # Call without target_columns to trigger auto-detection
        result = standardize_dates(data)
        
        # Should detect and convert birth_date
        assert pd.api.types.is_datetime64_any_dtype(result['birth_date'])
        # Should not convert other columns
        assert not pd.api.types.is_datetime64_any_dtype(result['patient_id'])
        assert not pd.api.types.is_datetime64_any_dtype(result['age'])
    
    def test_timeframe_validation(self):
        """Test timeframe validation functionality."""
        data = pd.DataFrame({
            'date_col': ['1990-01-15', '2025-01-15', '2023-01-15']  # Future date
        })
        
        result = standardize_dates(
            data, 
            target_columns=['date_col'],
            timeframe=('1900-01-01', '2024-12-31')
        )
        
        # Future date should be set to NaT
        assert pd.isna(result['date_col'].iloc[1])
        # Valid dates should remain
        assert not pd.isna(result['date_col'].iloc[0])
        assert not pd.isna(result['date_col'].iloc[2])
    
    def test_error_tolerance(self):
        """Test error tolerance handling."""
        data = pd.DataFrame({
            'date_col': ['1990-01-15', 'invalid_date', '2023-01-15', 'another_invalid']
        })
        
        # High tolerance - should process despite errors
        result_high = standardize_dates(
            data, 
            target_columns=['date_col'],
            error_tolerance=0.6  # 60% tolerance
        )
        assert pd.api.types.is_datetime64_any_dtype(result_high['date_col'])
        
        # Low tolerance - should skip processing
        result_low = standardize_dates(
            data, 
            target_columns=['date_col'],
            error_tolerance=0.1  # 10% tolerance
        )
        # Should remain as object type due to high error rate
        assert not pd.api.types.is_datetime64_any_dtype(result_low['date_col'])
    
    def test_already_datetime_columns(self):
        """Test handling of already datetime columns."""
        data = pd.DataFrame({
            'date_col': pd.to_datetime(['1990-01-15', '2023-01-15'])
        })
        
        result = standardize_dates(data, target_columns=['date_col'])
        
        # Should remain datetime and unchanged
        assert pd.api.types.is_datetime64_any_dtype(result['date_col'])
        pd.testing.assert_series_equal(data['date_col'], result['date_col'])
    
    def test_empty_data(self):
        """Test handling of empty data."""
        data = pd.DataFrame({
            'date_col': [None, None, None]
        })
        
        result = standardize_dates(data, target_columns=['date_col'])
        
        # Should handle gracefully
        assert len(result) == len(data)
    
    def test_mixed_valid_invalid_dates(self):
        """Test handling of mixed valid and invalid dates."""
        data = pd.DataFrame({
            'date_col': ['2023-01-15', 'not_a_date', '2023-02-20', None, '2023-03-10']
        })
        
        result = standardize_dates(data, target_columns=['date_col'])
        
        # Valid dates should be converted
        assert result['date_col'].iloc[0] == pd.Timestamp('2023-01-15')
        assert result['date_col'].iloc[2] == pd.Timestamp('2023-02-20')
        assert result['date_col'].iloc[4] == pd.Timestamp('2023-03-10')
        
        # Invalid dates should be NaT
        assert pd.isna(result['date_col'].iloc[1])
        assert pd.isna(result['date_col'].iloc[3])


class TestGetDefaultDateFormats:
    """Test default date formats functionality."""
    
    def test_format_list_not_empty(self):
        """Test that default formats list is not empty."""
        formats = get_default_date_formats()
        assert len(formats) > 0
        assert isinstance(formats, list)
    
    def test_common_formats_included(self):
        """Test that common formats are included."""
        formats = get_default_date_formats()
        
        # Check for some common formats
        assert '%Y-%m-%d' in formats
        assert '%d/%m/%Y' in formats
        assert '%m/%d/%Y' in formats


@pytest.fixture
def sample_date_data():
    """Sample data for testing."""
    return pd.DataFrame({
        'mixed_dates': ['2023-01-15', '15/02/2023', '2023/03/20', 'invalid'],
        'iso_dates': ['2023-01-15', '2023-02-20', '2023-03-25', 'invalid_date'],
        'text_col': ['text1', 'text2', 'text3', 'text4'],
        'numeric_col': [1, 2, 3, 4]
    })


def test_integration_date_standardization(sample_date_data):
    """Integration test for complete date standardization workflow."""
    # Test auto-detection and standardization
    result = standardize_dates(sample_date_data)
    
    # Should auto-detect and convert date columns
    date_columns = [col for col in result.columns 
                   if pd.api.types.is_datetime64_any_dtype(result[col])]
    
    assert len(date_columns) >= 1  # Should detect at least one date column
    
    # Non-date columns should remain unchanged
    assert result['text_col'].dtype == 'object'
    assert result['numeric_col'].dtype in ['int64', 'int32']