"""Tests for date sequence validation functionality."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from cleanepi.cleaning.date_sequence import (
    check_date_sequence,
    generate_date_sequence_report,
    detect_date_outliers,
    _check_single_sequence,
    _validate_sequences_by_subject,
    _validate_sequences_global
)


class TestDateSequenceValidation:
    """Test date sequence validation."""
    
    def test_valid_date_sequence(self):
        """Test validation of valid date sequences."""
        data = pd.DataFrame({
            'birth_date': pd.to_datetime(['1990-01-15', '1985-03-20', '1995-07-10']),
            'admission_date': pd.to_datetime(['2023-01-10', '2023-02-15', '2023-01-05']),
            'discharge_date': pd.to_datetime(['2023-01-20', '2023-02-25', '2023-01-15'])
        })
        
        result = check_date_sequence(
            data, 
            ['birth_date', 'admission_date', 'discharge_date']
        )
        
        assert 'date_sequence_valid' in result.columns
        assert 'date_sequence_issues' in result.columns
        # All sequences should be valid (birth < admission < discharge)
        assert result['date_sequence_valid'].all()
    
    def test_invalid_date_sequence(self):
        """Test detection of invalid date sequences."""
        data = pd.DataFrame({
            'birth_date': pd.to_datetime(['1990-01-15', '1985-03-20']),
            'admission_date': pd.to_datetime(['2023-01-10', '2023-02-15']),
            # Discharge before admission
            'discharge_date': pd.to_datetime(['2023-01-05', '2023-02-10'])
        })
        
        result = check_date_sequence(
            data, 
            ['birth_date', 'admission_date', 'discharge_date']
        )
        
        # Some sequences should be invalid
        assert not result['date_sequence_valid'].all()
        assert 'discharge_date_before_admission_date' in result['date_sequence_issues'].iloc[0]
    
    def test_with_subject_ids(self):
        """Test validation with subject ID grouping."""
        data = pd.DataFrame({
            'patient_id': ['P001', 'P002', 'P001'],
            'visit_1': pd.to_datetime(['2023-01-10', '2023-02-15', '2023-01-15']),
            'visit_2': pd.to_datetime(['2023-01-20', '2023-02-25', '2023-01-25'])
        })
        
        result = check_date_sequence(
            data,
            ['visit_1', 'visit_2'],
            subject_id_column='patient_id'
        )
        
        assert 'date_sequence_valid' in result.columns
        assert len(result) == 3  # Same number of rows
    
    def test_with_tolerance(self):
        """Test validation with date tolerance."""
        data = pd.DataFrame({
            'date1': pd.to_datetime(['2023-01-10', '2023-01-15']),
            'date2': pd.to_datetime(['2023-01-09', '2023-01-16'])  # One date slightly before
        })
        
        # With tolerance, should be valid
        result = check_date_sequence(
            data,
            ['date1', 'date2'],
            tolerance_days=2
        )
        
        assert result['date_sequence_valid'].all()
    
    def test_missing_dates(self):
        """Test handling of missing dates."""
        data = pd.DataFrame({
            'date1': pd.to_datetime(['2023-01-10', None, '2023-01-15']),
            'date2': pd.to_datetime(['2023-01-20', '2023-02-25', None])
        })
        
        result = check_date_sequence(data, ['date1', 'date2'])
        
        # Rows with missing dates should be invalid
        assert not result['date_sequence_valid'].iloc[1]
        assert not result['date_sequence_valid'].iloc[2]
        assert 'missing_dates' in result['date_sequence_issues'].iloc[1]
    
    def test_equal_dates_allowed(self):
        """Test allowing equal dates."""
        data = pd.DataFrame({
            'date1': pd.to_datetime(['2023-01-10', '2023-01-15']),
            'date2': pd.to_datetime(['2023-01-10', '2023-01-15'])  # Equal dates
        })
        
        result = check_date_sequence(
            data,
            ['date1', 'date2'],
            allow_equal=True
        )
        
        assert result['date_sequence_valid'].all()
    
    def test_equal_dates_not_allowed(self):
        """Test rejecting equal dates."""
        data = pd.DataFrame({
            'date1': pd.to_datetime(['2023-01-10']),
            'date2': pd.to_datetime(['2023-01-10'])  # Equal dates
        })
        
        result = check_date_sequence(
            data,
            ['date1', 'date2'],
            allow_equal=False
        )
        
        assert not result['date_sequence_valid'].iloc[0]
        assert 'date1_equals_date2' in result['date_sequence_issues'].iloc[0]


class TestDateSequenceHelpers:
    """Test helper functions for date sequence validation."""
    
    def test_check_single_sequence_valid(self):
        """Test single sequence validation with valid dates."""
        dates = [
            pd.Timestamp('1990-01-15'),
            pd.Timestamp('2023-01-10'),
            pd.Timestamp('2023-01-20')
        ]
        column_names = ['birth', 'admission', 'discharge']
        
        is_valid, issues = _check_single_sequence(dates, column_names, 0, True)
        
        assert is_valid
        assert len(issues) == 0
    
    def test_check_single_sequence_invalid(self):
        """Test single sequence validation with invalid dates."""
        dates = [
            pd.Timestamp('1990-01-15'),
            pd.Timestamp('2023-01-20'),
            pd.Timestamp('2023-01-10')  # Discharge before admission
        ]
        column_names = ['birth', 'admission', 'discharge']
        
        is_valid, issues = _check_single_sequence(dates, column_names, 0, True)
        
        assert not is_valid
        assert 'discharge_before_admission' in issues
    
    def test_future_birth_date(self):
        """Test detection of future birth dates."""
        future_date = pd.Timestamp.now() + pd.Timedelta(days=365)
        dates = [
            future_date,
            pd.Timestamp('2023-01-10'),
            pd.Timestamp('2023-01-20')
        ]
        column_names = ['birth', 'admission', 'discharge']
        
        is_valid, issues = _check_single_sequence(dates, column_names, 0, True)
        
        assert not is_valid
        assert 'future_birth_date' in issues
    
    def test_ancient_birth_date(self):
        """Test detection of ancient birth dates."""
        dates = [
            pd.Timestamp('1800-01-15'),  # Very old date
            pd.Timestamp('2023-01-10'),
            pd.Timestamp('2023-01-20')
        ]
        column_names = ['birth', 'admission', 'discharge']
        
        is_valid, issues = _check_single_sequence(dates, column_names, 0, True)
        
        assert not is_valid
        assert 'ancient_birth_date' in issues


class TestDateSequenceReporting:
    """Test date sequence reporting functionality."""
    
    def test_generate_report(self):
        """Test report generation."""
        data = pd.DataFrame({
            'birth_date': pd.to_datetime(['1990-01-15', '1985-03-20']),
            'admission_date': pd.to_datetime(['2023-01-10', '2023-02-15']),
            'discharge_date': pd.to_datetime(['2023-01-05', '2023-02-25']),
            'date_sequence_valid': [False, True],
            'date_sequence_issues': ['discharge_date_before_admission_date', '']
        })
        
        report = generate_date_sequence_report(
            data, 
            ['birth_date', 'admission_date', 'discharge_date']
        )
        
        assert 'summary' in report
        assert 'column_analysis' in report
        assert 'issue_breakdown' in report
        assert 'recommendations' in report
        
        assert report['summary']['total_sequences'] == 2
        assert report['summary']['valid_sequences'] == 1
        assert report['summary']['invalid_sequences'] == 1


class TestDateOutlierDetection:
    """Test date outlier detection functionality."""
    
    def test_detect_outliers_iqr(self):
        """Test outlier detection using IQR method."""
        # Create data with one clear outlier
        normal_dates = pd.date_range('2023-01-01', periods=10, freq='D')
        outlier_date = pd.Timestamp('1990-01-01')
        
        data = pd.DataFrame({
            'dates': list(normal_dates) + [outlier_date]
        })
        
        result = detect_date_outliers(data, ['dates'], method='iqr')
        
        assert 'dates_outlier' in result.columns
        # The 1990 date should be detected as an outlier
        assert result['dates_outlier'].iloc[-1] == True
    
    def test_detect_outliers_std(self):
        """Test outlier detection using standard deviation method."""
        # Create data with one clear outlier
        normal_dates = pd.date_range('2023-01-01', periods=10, freq='D')
        outlier_date = pd.Timestamp('1990-01-01')
        
        data = pd.DataFrame({
            'dates': list(normal_dates) + [outlier_date]
        })
        
        result = detect_date_outliers(data, ['dates'], method='std')
        
        assert 'dates_outlier' in result.columns
        # The 1990 date should be detected as an outlier
        assert result['dates_outlier'].iloc[-1] == True
    
    def test_detect_outliers_percentile(self):
        """Test outlier detection using percentile method."""
        # Create data with one clear outlier
        normal_dates = pd.date_range('2023-01-01', periods=10, freq='D')
        outlier_date = pd.Timestamp('1990-01-01')
        
        data = pd.DataFrame({
            'dates': list(normal_dates) + [outlier_date]
        })
        
        result = detect_date_outliers(data, ['dates'], method='percentile')
        
        assert 'dates_outlier' in result.columns
    
    def test_invalid_outlier_method(self):
        """Test handling of invalid outlier detection method."""
        data = pd.DataFrame({
            'dates': pd.date_range('2023-01-01', periods=5, freq='D')
        })
        
        result = detect_date_outliers(data, ['dates'], method='invalid')
        
        # Should not add outlier column for invalid method
        assert 'dates_outlier' not in result.columns


class TestEdgeCases:
    """Test edge cases for date sequence validation."""
    
    def test_single_date_column(self):
        """Test with only one date column."""
        data = pd.DataFrame({
            'date1': pd.to_datetime(['2023-01-10', '2023-01-15'])
        })
        
        result = check_date_sequence(data, ['date1'])
        
        # Should return original data since sequence validation needs 2+ columns
        assert result.equals(data)
    
    def test_non_datetime_columns(self):
        """Test with non-datetime columns."""
        data = pd.DataFrame({
            'date1': ['2023-01-10', '2023-01-15'],  # String dates
            'date2': ['2023-01-20', '2023-01-25']
        })
        
        result = check_date_sequence(data, ['date1', 'date2'])
        
        # Should convert to datetime and validate
        assert 'date_sequence_valid' in result.columns
    
    def test_invalid_subject_id_column(self):
        """Test with invalid subject ID column."""
        data = pd.DataFrame({
            'date1': pd.to_datetime(['2023-01-10', '2023-01-15']),
            'date2': pd.to_datetime(['2023-01-20', '2023-01-25'])
        })
        
        result = check_date_sequence(
            data,
            ['date1', 'date2'],
            subject_id_column='nonexistent_column'
        )
        
        # Should ignore invalid subject ID column and validate globally
        assert 'date_sequence_valid' in result.columns