"""Tests for dictionary-based cleaning functionality."""

import pytest
import pandas as pd
import numpy as np

from cleanepi.cleaning.dictionary_cleaning import (
    clean_using_dictionary,
    create_mapping_dictionary,
    validate_dictionary_mappings,
    _apply_column_mapping,
    _find_partial_match
)


class TestDictionaryCleaning:
    """Test dictionary-based cleaning functionality."""
    
    def test_basic_dictionary_cleaning(self):
        """Test basic dictionary cleaning."""
        data = pd.DataFrame({
            'status': ['pos', 'neg', 'positive', 'negative'],
            'gender': ['M', 'F', 'm', 'f']
        })
        
        dictionary = {
            'status': {'pos': 'positive', 'neg': 'negative'},
            'gender': {'M': 'Male', 'F': 'Female', 'm': 'Male', 'f': 'Female'}
        }
        
        result = clean_using_dictionary(data, dictionary)
        
        # Check that mappings were applied
        assert result['status'].iloc[0] == 'positive'  # pos -> positive
        assert result['status'].iloc[1] == 'negative'  # neg -> negative
        assert result['status'].iloc[2] == 'positive'  # already positive
        assert result['gender'].iloc[0] == 'Male'      # M -> Male
        assert result['gender'].iloc[3] == 'Female'    # f -> Female
    
    def test_case_sensitive_mapping(self):
        """Test case-sensitive mapping."""
        data = pd.DataFrame({
            'status': ['POS', 'pos', 'Pos']
        })
        
        dictionary = {
            'status': {'pos': 'positive'}
        }
        
        # Case sensitive (default)
        result_sensitive = clean_using_dictionary(data, dictionary, case_sensitive=True)
        assert result_sensitive['status'].iloc[0] == 'POS'  # Not mapped
        assert result_sensitive['status'].iloc[1] == 'positive'  # Mapped
        
        # Case insensitive
        result_insensitive = clean_using_dictionary(data, dictionary, case_sensitive=False)
        assert result_insensitive['status'].iloc[0] == 'positive'  # Mapped
        assert result_insensitive['status'].iloc[1] == 'positive'  # Mapped
        assert result_insensitive['status'].iloc[2] == 'positive'  # Mapped
    
    def test_partial_matching(self):
        """Test partial matching functionality."""
        data = pd.DataFrame({
            'result': ['positive result', 'negative test', 'pos', 'neg']
        })
        
        dictionary = {
            'result': {'positive': 'pos', 'negative': 'neg'}
        }
        
        # Exact match only (default)
        result_exact = clean_using_dictionary(data, dictionary, exact_match=True)
        assert result_exact['result'].iloc[0] == 'positive result'  # Not mapped
        
        # Partial matching
        result_partial = clean_using_dictionary(data, dictionary, exact_match=False)
        assert result_partial['result'].iloc[0] == 'pos'  # Mapped (contains 'positive')
        assert result_partial['result'].iloc[1] == 'neg'  # Mapped (contains 'negative')
    
    def test_default_actions(self):
        """Test different default actions for unmapped values."""
        data = pd.DataFrame({
            'status': ['pos', 'unknown', 'neg']
        })
        
        dictionary = {
            'status': {'pos': 'positive', 'neg': 'negative'}
        }
        
        # Keep unmapped values (default)
        result_keep = clean_using_dictionary(data, dictionary, default_action='keep')
        assert result_keep['status'].iloc[1] == 'unknown'
        
        # Convert unmapped to null
        result_null = clean_using_dictionary(data, dictionary, default_action='null')
        assert pd.isna(result_null['status'].iloc[1])
        
        # Flag unmapped values
        result_flag = clean_using_dictionary(data, dictionary, default_action='flag')
        assert result_flag['status'].iloc[1] == 'unknown_unmapped'
    
    def test_missing_columns(self):
        """Test handling of missing columns."""
        data = pd.DataFrame({
            'existing_col': ['a', 'b', 'c']
        })
        
        dictionary = {
            'existing_col': {'a': 'A'}
        }
        
        # Should process existing columns
        result = clean_using_dictionary(data, dictionary)
        assert 'existing_col' in result.columns
        assert result['existing_col'].iloc[0] == 'A'
    
    def test_null_values(self):
        """Test handling of null values."""
        data = pd.DataFrame({
            'status': ['pos', None, 'neg', np.nan]
        })
        
        dictionary = {
            'status': {'pos': 'positive', 'neg': 'negative'}
        }
        
        result = clean_using_dictionary(data, dictionary)
        
        # Null values should remain null
        assert pd.isna(result['status'].iloc[1])
        assert pd.isna(result['status'].iloc[3])
        # Non-null values should be mapped
        assert result['status'].iloc[0] == 'positive'
        assert result['status'].iloc[2] == 'negative'


class TestMappingHelpers:
    """Test helper functions for dictionary mapping."""
    
    def test_apply_column_mapping(self):
        """Test applying mapping to a single column."""
        series = pd.Series(['a', 'b', 'c', 'a'])
        mappings = {'a': 'A', 'b': 'B'}
        
        result_series, stats = _apply_column_mapping(
            series, mappings, case_sensitive=True, exact_match=True, default_action='keep'
        )
        
        assert result_series.iloc[0] == 'A'
        assert result_series.iloc[1] == 'B'
        assert result_series.iloc[2] == 'c'  # Unmapped
        assert result_series.iloc[3] == 'A'
        
        assert stats['total_values'] == 4
        assert stats['mapped_values'] == 3  # 2 'a's + 1 'b'
        assert stats['unmapped_values'] == 1  # 1 'c'
    
    def test_find_partial_match(self):
        """Test partial matching logic."""
        mappings = {'positive': 'pos', 'negative': 'neg'}
        
        # Should find match
        result = _find_partial_match('positive result', mappings, case_sensitive=False)
        assert result == 'pos'
        
        # Should not find match
        result = _find_partial_match('unknown', mappings, case_sensitive=False)
        assert result is None
        
        # Case sensitivity
        result_sensitive = _find_partial_match('POSITIVE', mappings, case_sensitive=True)
        assert result_sensitive is None
        
        result_insensitive = _find_partial_match('POSITIVE', mappings, case_sensitive=False)
        assert result_insensitive == 'pos'


class TestMappingDictionaryCreation:
    """Test mapping dictionary creation functionality."""
    
    def test_create_mapping_dictionary(self):
        """Test creating mapping dictionary from data."""
        data = pd.DataFrame({
            'status': ['pos', 'neg', 'pos', 'positive'],
            'gender': ['M', 'F', 'M', 'F']
        })
        
        template = create_mapping_dictionary(data, ['status', 'gender'])
        
        assert 'status' in template
        assert 'gender' in template
        
        # Check status mappings
        status_mappings = template['status']
        assert 'pos' in status_mappings
        assert 'neg' in status_mappings
        assert 'positive' in status_mappings
        
        # Check counts
        assert status_mappings['pos']['count'] == 2
        assert status_mappings['neg']['count'] == 1
        assert status_mappings['positive']['count'] == 1
    
    def test_create_mapping_dictionary_without_counts(self):
        """Test creating mapping dictionary without counts."""
        data = pd.DataFrame({
            'status': ['pos', 'neg', 'pos']
        })
        
        template = create_mapping_dictionary(data, ['status'], include_counts=False)
        
        # Should be simple string mappings
        assert template['status']['pos'] == 'pos'
        assert template['status']['neg'] == 'neg'


class TestMappingValidation:
    """Test mapping validation functionality."""
    
    def test_validate_dictionary_mappings(self):
        """Test validation of dictionary mappings."""
        data = pd.DataFrame({
            'status': ['pos', 'neg', 'positive', 'unknown'],
            'gender': ['M', 'F', 'Other', 'M']
        })
        
        dictionary = {
            'status': {'pos': 'positive', 'neg': 'negative', 'positive': 'positive'},
            'gender': {'M': 'Male', 'F': 'Female'},  # Missing 'Other'
        }
        
        report = validate_dictionary_mappings(data, dictionary)
        
        assert 'columns' in report
        assert 'summary' in report
        
        # Status should have good coverage
        status_report = report['columns']['status']
        assert status_report['coverage_rate'] >= 0.75
        
        # Gender should have warning due to missing mapping
        gender_report = report['columns']['gender']
        assert 'Other' in gender_report['missing_values']


class TestEdgeCases:
    """Test edge cases for dictionary cleaning."""
    
    def test_empty_dictionary(self):
        """Test with empty dictionary."""
        data = pd.DataFrame({
            'status': ['pos', 'neg']
        })
        
        result = clean_using_dictionary(data, {})
        
        # Should return original data unchanged
        pd.testing.assert_frame_equal(result, data)
    
    def test_empty_data(self):
        """Test with empty data."""
        data = pd.DataFrame({
            'status': []
        })
        
        dictionary = {
            'status': {'pos': 'positive'}
        }
        
        with pytest.raises(ValueError, match="DataFrame is empty"):
            clean_using_dictionary(data, dictionary)
    
    def test_invalid_dictionary_structure(self):
        """Test with invalid dictionary structure."""
        data = pd.DataFrame({
            'status': ['pos', 'neg']
        })
        
        # Dictionary is not a dict
        with pytest.raises(ValueError, match="dictionary must be a dict"):
            clean_using_dictionary(data, "not_a_dict")
        
        # Column mapping is not a dict
        with pytest.raises(ValueError, match="Mappings for column .* must be a dict"):
            clean_using_dictionary(data, {'status': 'not_a_dict'})
    
    def test_numeric_values(self):
        """Test with numeric values."""
        data = pd.DataFrame({
            'code': [1, 2, 3, 1]
        })
        
        dictionary = {
            'code': {'1': 'positive', '2': 'negative'}
        }
        
        result = clean_using_dictionary(data, dictionary)
        
        # Numeric values should be converted to strings for matching
        assert result['code'].iloc[0] == 'positive'
        assert result['code'].iloc[1] == 'negative'
        assert result['code'].iloc[2] == 3  # Unmapped, kept as is
        assert result['code'].iloc[3] == 'positive'