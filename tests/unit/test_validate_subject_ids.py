"""Tests for subject ID validation functionality."""

import pytest
import pandas as pd
import numpy as np

from cleanepi.cleaning.validate_subject_ids import (
    check_subject_ids,
    generate_subject_id_report,
    _validate_ids,
    _extract_numeric_part,
    _check_cross_column_duplicates
)


class TestBasicSubjectIDValidation:
    """Test basic subject ID validation functionality."""
    
    def test_prefix_validation(self):
        """Test validation with prefix requirements."""
        data = pd.DataFrame({
            'patient_id': ['P001', 'P002', 'X003', 'P004']
        })
        
        result = check_subject_ids(
            data,
            target_columns=['patient_id'],
            prefix='P'
        )
        
        # Check validation columns are added
        assert 'patient_id_valid' in result.columns
        assert 'patient_id_issues' in result.columns
        
        # Check validation results
        assert result['patient_id_valid'].iloc[0] == True  # P001
        assert result['patient_id_valid'].iloc[1] == True  # P002
        assert result['patient_id_valid'].iloc[2] == False  # X003 (wrong prefix)
        assert result['patient_id_valid'].iloc[3] == True  # P004
        
        # Check issues
        assert 'missing_prefix_P' in result['patient_id_issues'].iloc[2]
    
    def test_suffix_validation(self):
        """Test validation with suffix requirements.""" 
        data = pd.DataFrame({
            'study_id': ['001A', '002A', '003B', '004A']
        })
        
        result = check_subject_ids(
            data,
            target_columns=['study_id'],
            suffix='A'
        )
        
        assert result['study_id_valid'].iloc[0] == True  # 001A
        assert result['study_id_valid'].iloc[1] == True  # 002A
        assert result['study_id_valid'].iloc[2] == False  # 003B (wrong suffix)
        assert result['study_id_valid'].iloc[3] == True  # 004A
    
    def test_character_length_validation(self):
        """Test validation with character length requirements."""
        data = pd.DataFrame({
            'id_col': ['001', '0001', '12', '1234']
        })
        
        result = check_subject_ids(
            data,
            target_columns=['id_col'],
            nchar=4
        )
        
        assert result['id_col_valid'].iloc[0] == False  # '001' (3 chars)
        assert result['id_col_valid'].iloc[1] == True   # '0001' (4 chars)
        assert result['id_col_valid'].iloc[2] == False  # '12' (2 chars) 
        assert result['id_col_valid'].iloc[3] == True   # '1234' (4 chars)
    
    def test_numeric_range_validation(self):
        """Test validation with numeric range requirements."""
        data = pd.DataFrame({
            'patient_id': ['P001', 'P150', 'P999', 'P1500']
        })
        
        result = check_subject_ids(
            data,
            target_columns=['patient_id'],
            prefix='P',
            range=(1, 1000)
        )
        
        assert result['patient_id_valid'].iloc[0] == True   # P001 (1 in range)
        assert result['patient_id_valid'].iloc[1] == True   # P150 (150 in range)
        assert result['patient_id_valid'].iloc[2] == True   # P999 (999 in range)
        assert result['patient_id_valid'].iloc[3] == False  # P1500 (1500 out of range)
    
    def test_custom_pattern_validation(self):
        """Test validation with custom regex patterns."""
        data = pd.DataFrame({
            'complex_id': ['ABC123', 'XYZ456', '123ABC', 'ABC12']
        })
        
        result = check_subject_ids(
            data,
            target_columns=['complex_id'],
            pattern=r'^[A-Z]{3}\d{3}$'  # Three letters followed by three digits
        )
        
        assert result['complex_id_valid'].iloc[0] == True   # ABC123 (matches)
        assert result['complex_id_valid'].iloc[1] == True   # XYZ456 (matches)
        assert result['complex_id_valid'].iloc[2] == False  # 123ABC (doesn't match)
        assert result['complex_id_valid'].iloc[3] == False  # ABC12 (doesn't match)


class TestCombinedValidation:
    """Test validation with multiple criteria."""
    
    def test_prefix_and_length(self):
        """Test validation with both prefix and length requirements."""
        data = pd.DataFrame({
            'patient_id': ['P001', 'P12', 'X001', 'P1234']
        })
        
        result = check_subject_ids(
            data,
            target_columns=['patient_id'],
            prefix='P',
            nchar=4
        )
        
        assert result['patient_id_valid'].iloc[0] == True   # P001 (correct prefix and length)
        assert result['patient_id_valid'].iloc[1] == False  # P12 (correct prefix, wrong length)
        assert result['patient_id_valid'].iloc[2] == False  # X001 (wrong prefix, correct length)
        assert result['patient_id_valid'].iloc[3] == False  # P1234 (correct prefix, wrong length)
    
    def test_all_criteria(self):
        """Test validation with all criteria combined."""
        data = pd.DataFrame({
            'id_col': ['P001S', 'P002S', 'P999S', 'X001S', 'P1001S']
        })
        
        result = check_subject_ids(
            data,
            target_columns=['id_col'],
            prefix='P',
            suffix='S',
            nchar=5,
            range=(1, 1000)
        )
        
        assert result['id_col_valid'].iloc[0] == True   # P001S (all criteria met)
        assert result['id_col_valid'].iloc[1] == True   # P002S (all criteria met)
        assert result['id_col_valid'].iloc[2] == True   # P999S (all criteria met)
        assert result['id_col_valid'].iloc[3] == False  # X001S (wrong prefix)
        assert result['id_col_valid'].iloc[4] == False  # P1001S (range violation and length)


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_extract_numeric_part(self):
        """Test numeric part extraction."""
        test_cases = [
            ('P001', 'P', None, '001'),
            ('P123S', 'P', 'S', '123'),
            ('ABC456XYZ', 'ABC', 'XYZ', '456'),
            ('123', None, None, '123'),
            ('NONUM', None, None, None)
        ]
        
        for id_str, prefix, suffix, expected in test_cases:
            result = _extract_numeric_part(id_str, prefix, suffix)
            assert result == expected
    
    def test_validate_ids_function(self):
        """Test the _validate_ids helper function."""
        ids = pd.Series(['P001', 'P002', 'X003', 'P999'])
        
        results = _validate_ids(
            ids,
            prefix='P',
            nchar=4,
            range=(1, 500)
        )
        
        assert results['P001']['valid'] == True
        assert results['P002']['valid'] == True
        assert results['X003']['valid'] == False
        assert results['P999']['valid'] == False  # Out of range
        
        # Check issue messages
        assert 'missing_prefix_P' in results['X003']['issues']
        assert 'range_violation' in str(results['P999']['issues'])
    
    def test_cross_column_duplicates(self):
        """Test cross-column duplicate detection."""
        data = pd.DataFrame({
            'col1': ['A001', 'A002', 'A003'],
            'col2': ['B001', 'A002', 'B003'],  # A002 appears in both columns
            'col3': ['C001', 'C002', 'C003']
        })
        
        duplicates = _check_cross_column_duplicates(data, ['col1', 'col2', 'col3'])
        
        assert 'A002' in duplicates
        assert 'col1' in duplicates['A002']
        assert 'col2' in duplicates['A002']
        assert len(duplicates['A002']) == 2


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_data(self):
        """Test handling of empty data."""
        data = pd.DataFrame({
            'patient_id': []
        })
        
        # Should raise validation error for empty DataFrame
        with pytest.raises(ValueError, match="DataFrame is empty"):
            check_subject_ids(
                data,
                target_columns=['patient_id'],
                prefix='P'
            )
    
    def test_null_values(self):
        """Test handling of null values."""
        data = pd.DataFrame({
            'patient_id': ['P001', None, 'P003', '']
        })
        
        result = check_subject_ids(
            data,
            target_columns=['patient_id'],
            prefix='P'
        )
        
        # Null values should be marked as invalid
        assert result['patient_id_valid'].iloc[0] == True   # P001
        assert result['patient_id_valid'].iloc[1] == False  # None
        assert result['patient_id_valid'].iloc[2] == True   # P003
        assert result['patient_id_valid'].iloc[3] == False  # Empty string
        
        # Check that null handling is in issues
        assert 'missing_value' in result['patient_id_issues'].iloc[1]
    
    def test_invalid_regex_pattern(self):
        """Test handling of invalid regex patterns."""
        data = pd.DataFrame({
            'id_col': ['ABC123', 'DEF456']
        })
        
        # This should handle invalid regex gracefully
        result = check_subject_ids(
            data,
            target_columns=['id_col'],
            pattern='[invalid regex'  # Invalid regex
        )
        
        # Should mark all as invalid due to regex error
        assert result['id_col_valid'].iloc[0] == False
        assert result['id_col_valid'].iloc[1] == False
    
    def test_multiple_columns(self):
        """Test validation of multiple columns."""
        data = pd.DataFrame({
            'patient_id': ['P001', 'P002', 'X003'],
            'study_id': ['S001', 'S002', 'S003']
        })
        
        result = check_subject_ids(
            data,
            target_columns=['patient_id', 'study_id'],
            prefix='P'  # Only applies to both columns
        )
        
        # patient_id should validate correctly
        assert result['patient_id_valid'].iloc[0] == True
        assert result['patient_id_valid'].iloc[1] == True
        assert result['patient_id_valid'].iloc[2] == False
        
        # study_id should all fail (wrong prefix)
        assert result['study_id_valid'].iloc[0] == False
        assert result['study_id_valid'].iloc[1] == False
        assert result['study_id_valid'].iloc[2] == False


class TestReportGeneration:
    """Test report generation functionality."""
    
    def test_subject_id_report(self):
        """Test generation of subject ID validation report."""
        data = pd.DataFrame({
            'patient_id': ['P001', 'P002', 'X003', 'P004']
        })
        
        # First validate
        validated_data = check_subject_ids(
            data,
            target_columns=['patient_id'],
            prefix='P'
        )
        
        # Then generate report
        report = generate_subject_id_report(validated_data, ['patient_id'])
        
        # Check report structure
        assert 'columns' in report
        assert 'summary' in report
        assert 'patient_id' in report['columns']
        
        # Check summary statistics
        col_stats = report['columns']['patient_id']
        assert col_stats['total_ids'] == 4
        assert col_stats['valid_ids'] == 3  # P001, P002, P004
        assert col_stats['invalid_ids'] == 1  # X003
        assert col_stats['valid_percentage'] == 75.0
        
        # Check overall summary
        summary = report['summary']
        assert summary['total_ids'] == 4
        assert summary['valid_ids'] == 3
        assert summary['columns_processed'] == 1


@pytest.fixture
def sample_id_data():
    """Sample data for testing."""
    return pd.DataFrame({
        'patient_id': ['P001', 'P002', 'P003', 'X004', 'P999'],
        'study_id': ['S001', 'S002', 'S003', 'S004', 'S005'],
        'visit_id': ['V001', 'V002', None, 'V004', 'invalid']
    })


def test_integration_subject_id_validation(sample_id_data):
    """Integration test for complete subject ID validation."""
    result = check_subject_ids(
        sample_id_data,
        target_columns=['patient_id', 'study_id'],
        prefix='P',  # Will only match patient_id
        nchar=4
    )
    
    # Should add validation columns for both target columns
    assert 'patient_id_valid' in result.columns
    assert 'patient_id_issues' in result.columns
    assert 'study_id_valid' in result.columns
    assert 'study_id_issues' in result.columns
    
    # Generate and check report
    report = generate_subject_id_report(result, ['patient_id', 'study_id'])
    assert len(report['columns']) == 2
    assert report['summary']['columns_processed'] == 2