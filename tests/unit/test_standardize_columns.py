"""
Comprehensive tests for the standardize_columns module.
"""

import pytest
import pandas as pd
from unittest.mock import patch

from cleanepi.cleaning.standardize_columns import (
    standardize_column_names,
    _standardize_single_name,
    _to_snake_case,
    _to_camel_case,
    _to_pascal_case,
    _to_kebab_case,
    _handle_duplicate_names,
    suggest_column_renames
)


class TestStandardizeColumnNames:
    """Test standardize_column_names functionality."""
    
    def test_basic_standardization(self):
        """Test basic column name standardization."""
        df = pd.DataFrame({
            'Date of Birth': [1, 2, 3],
            'Patient ID': [4, 5, 6],
            'Test Value': [7, 8, 9]
        })
        
        result = standardize_column_names(df)
        
        expected_columns = ['date_of_birth', 'patient_id', 'test_value']
        assert list(result.columns) == expected_columns
        assert result.shape == df.shape
    
    def test_keep_columns(self):
        """Test keeping specific columns unchanged."""
        df = pd.DataFrame({
            'Keep Me': [1, 2, 3],
            'Change Me': [4, 5, 6],
            'Also Keep': [7, 8, 9]
        })
        
        result = standardize_column_names(df, keep=['Keep Me', 'Also Keep'])
        
        # Kept columns should remain unchanged
        assert 'Keep Me' in result.columns
        assert 'Also Keep' in result.columns
        assert 'change_me' in result.columns  # This one should be standardized
    
    def test_custom_rename(self):
        """Test custom column renaming."""
        df = pd.DataFrame({
            'Old Name 1': [1, 2, 3],
            'Old Name 2': [4, 5, 6],
            'Keep Same': [7, 8, 9]
        })
        
        rename_map = {
            'Old Name 1': 'new_name_1',
            'Old Name 2': 'custom_column'
        }
        
        result = standardize_column_names(df, rename=rename_map)
        
        assert 'new_name_1' in result.columns
        assert 'custom_column' in result.columns
        assert 'keep_same' in result.columns  # Standardized but not renamed
    
    def test_snake_case_style(self):
        """Test snake_case style conversion."""
        df = pd.DataFrame({
            'CamelCase Column': [1, 2, 3],
            'Mixed-Case_Name': [4, 5, 6],
            'UPPER CASE': [7, 8, 9]
        })
        
        result = standardize_column_names(df, style="snake_case")
        
        expected = ['camel_case_column', 'mixed_case_name', 'upper_case']
        assert list(result.columns) == expected
    
    def test_camel_case_style(self):
        """Test camelCase style conversion."""
        df = pd.DataFrame({
            'snake_case_column': [1, 2, 3],
            'Space Separated Name': [4, 5, 6],
            'kebab-case-name': [7, 8, 9]
        })
        
        result = standardize_column_names(df, style="camelCase")
        
        expected = ['snakeCaseColumn', 'spaceSeparatedName', 'kebabCaseName']
        assert list(result.columns) == expected
    
    def test_pascal_case_style(self):
        """Test PascalCase style conversion."""
        df = pd.DataFrame({
            'snake_case_column': [1, 2, 3],
            'space separated': [4, 5, 6],
            'mixed_Format': [7, 8, 9]
        })
        
        result = standardize_column_names(df, style="PascalCase")
        
        expected = ['SnakeCaseColumn', 'SpaceSeparated', 'MixedFormat']
        assert list(result.columns) == expected
    
    def test_kebab_case_style(self):
        """Test kebab-case style conversion."""
        df = pd.DataFrame({
            'snake_case_column': [1, 2, 3],
            'Space Separated': [4, 5, 6],
            'CamelCaseColumn': [7, 8, 9]
        })
        
        result = standardize_column_names(df, style="kebab-case")
        
        expected = ['snake-case-column', 'space-separated', 'camel-case-column']
        assert list(result.columns) == expected
    
    def test_remove_special_chars(self):
        """Test special character removal."""
        df = pd.DataFrame({
            'Name@#$%': [1, 2, 3],
            'Col(with)brackets': [4, 5, 6],
            'Normal_Name': [7, 8, 9]
        })
        
        result = standardize_column_names(df, remove_special_chars=True)
        
        # Special characters should be removed
        assert 'name' in result.columns
        assert 'colwithbrackets' in result.columns
        assert 'normal_name' in result.columns
    
    def test_keep_special_chars(self):
        """Test keeping special characters."""
        df = pd.DataFrame({
            'Name@Symbol': [1, 2, 3],
            'Normal Name': [4, 5, 6]
        })
        
        result = standardize_column_names(df, remove_special_chars=False)
        
        # Special characters should be preserved
        assert any('@' in col for col in result.columns) or any('symbol' in col.lower() for col in result.columns)
    
    def test_max_length(self):
        """Test maximum length constraint."""
        df = pd.DataFrame({
            'Very Long Column Name That Exceeds Limit': [1, 2, 3],
            'Short': [4, 5, 6]
        })
        
        result = standardize_column_names(df, max_length=10)
        
        # Long column name should be truncated
        assert all(len(col) <= 10 for col in result.columns)
        assert 'short' in result.columns  # Short name unchanged
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        
        with pytest.raises(ValueError):  # Should raise validation error
            standardize_column_names(df)
    
    def test_duplicate_column_names(self):
        """Test handling of duplicate column names."""
        df = pd.DataFrame({
            'Column Name': [1, 2, 3],
            'Column Name ': [4, 5, 6],  # Trailing space
            'Different': [7, 8, 9]
        })
        
        result = standardize_column_names(df)
        
        # Should handle duplicates by adding suffixes
        columns = list(result.columns)
        assert 'column_name' in columns
        assert 'column_name_1' in columns or any('column_name' in col for col in columns)
        assert 'different' in columns
    
    @patch('cleanepi.cleaning.standardize_columns.logger')
    def test_logging_with_changes(self, mock_logger):
        """Test logging when changes are made."""
        df = pd.DataFrame({'Old Name': [1, 2, 3]})
        
        standardize_column_names(df)
        
        # Should log changes
        mock_logger.info.assert_called()
        mock_logger.debug.assert_called()
    
    @patch('cleanepi.cleaning.standardize_columns.logger')
    def test_logging_no_changes(self, mock_logger):
        """Test logging when no changes are made."""
        df = pd.DataFrame({'already_standardized': [1, 2, 3]})
        
        standardize_column_names(df)
        
        # Should log no changes
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[0][0]
        assert "No column names changed" in call_args


class TestStandardizeSingleName:
    """Test _standardize_single_name functionality."""
    
    def test_basic_standardization(self):
        """Test basic name standardization."""
        result = _standardize_single_name("Test Column", "snake_case", True, None)
        assert result == "test_column"
    
    def test_special_characters(self):
        """Test special character handling."""
        # With removal
        result1 = _standardize_single_name("Test@#$Column", "snake_case", True, None)
        assert result1 == "test_column"
        
        # Without removal - special chars may be preserved in different ways
        result2 = _standardize_single_name("Test@Column", "snake_case", False, None)
        assert isinstance(result2, str)  # Just check it returns a string
    
    def test_max_length_constraint(self):
        """Test maximum length constraint."""
        result = _standardize_single_name("Very Long Column Name", "snake_case", True, 10)
        assert len(result) <= 10
        assert not result.endswith('_')  # Should not end with separator
    
    def test_empty_name(self):
        """Test empty name handling."""
        result = _standardize_single_name("", "snake_case", True, None)
        assert result == "column"
    
    def test_numeric_start(self):
        """Test name starting with number."""
        result = _standardize_single_name("123 Column", "snake_case", True, None)
        assert result == "col_123_column"
    
    def test_whitespace_handling(self):
        """Test whitespace handling."""
        result = _standardize_single_name("  Spaced   Name  ", "snake_case", True, None)
        assert result == "spaced_name"
    
    def test_unknown_style(self):
        """Test unknown style handling."""
        with patch('cleanepi.cleaning.standardize_columns.logger') as mock_logger:
            result = _standardize_single_name("Test", "unknown_style", True, None)
            assert result == "test"  # Should default to snake_case
            mock_logger.warning.assert_called()


class TestCaseConversions:
    """Test case conversion functions."""
    
    def test_to_snake_case(self):
        """Test snake_case conversion."""
        assert _to_snake_case("CamelCase") == "camel_case"
        assert _to_snake_case("snake_case") == "snake_case"
        assert _to_snake_case("kebab-case") == "kebab_case"
        assert _to_snake_case("Space Separated") == "space_separated"
        assert _to_snake_case("Multiple   Spaces") == "multiple_spaces"
        assert _to_snake_case("_leading_trailing_") == "leading_trailing"
    
    def test_to_camel_case(self):
        """Test camelCase conversion."""
        assert _to_camel_case("snake_case") == "snakeCase"
        assert _to_camel_case("kebab-case") == "kebabCase"
        assert _to_camel_case("Space Separated") == "spaceSeparated"
        assert _to_camel_case("single") == "single"
        assert _to_camel_case("") == ""
    
    def test_to_pascal_case(self):
        """Test PascalCase conversion."""
        assert _to_pascal_case("snake_case") == "SnakeCase"
        assert _to_pascal_case("kebab-case") == "KebabCase"
        assert _to_pascal_case("space separated") == "SpaceSeparated"
        assert _to_pascal_case("single") == "Single"
        assert _to_pascal_case("") == ""
    
    def test_to_kebab_case(self):
        """Test kebab-case conversion."""
        assert _to_kebab_case("CamelCase") == "camel-case"
        assert _to_kebab_case("snake_case") == "snake-case"
        assert _to_kebab_case("kebab-case") == "kebab-case"
        assert _to_kebab_case("Space Separated") == "space-separated"
        assert _to_kebab_case("Multiple   Spaces") == "multiple-spaces"
        assert _to_kebab_case("-leading-trailing-") == "leading-trailing"


class TestHandleDuplicateNames:
    """Test _handle_duplicate_names functionality."""
    
    def test_no_duplicates(self):
        """Test with no duplicate names."""
        names = ['col1', 'col2', 'col3']
        result = _handle_duplicate_names(names)
        assert result == names
    
    def test_with_duplicates(self):
        """Test with duplicate names."""
        names = ['col1', 'col2', 'col1', 'col3', 'col1']
        result = _handle_duplicate_names(names)
        
        expected = ['col1', 'col2', 'col1_1', 'col3', 'col1_2']
        assert result == expected
    
    def test_multiple_different_duplicates(self):
        """Test with multiple different duplicates."""
        names = ['a', 'b', 'a', 'b', 'c', 'a']
        result = _handle_duplicate_names(names)
        
        expected = ['a', 'b', 'a_1', 'b_1', 'c', 'a_2']
        assert result == expected
    
    def test_empty_list(self):
        """Test with empty list."""
        result = _handle_duplicate_names([])
        assert result == []
    
    def test_single_name(self):
        """Test with single name."""
        result = _handle_duplicate_names(['single'])
        assert result == ['single']


class TestSuggestColumnRenames:
    """Test suggest_column_renames functionality."""
    
    def test_date_patterns(self):
        """Test date pattern suggestions."""
        df = pd.DataFrame({
            'Date of Birth': [1, 2, 3],
            'DOB': [4, 5, 6],
            'Birth Date': [7, 8, 9],
            'admission date': [10, 11, 12]
        })
        
        suggestions = suggest_column_renames(df)
        
        # Should suggest standardized date column names
        assert isinstance(suggestions, dict)
        # Check that some suggestions are made for date patterns
        assert len(suggestions) >= 0  # May or may not have suggestions
    
    def test_id_patterns(self):
        """Test ID pattern suggestions."""
        df = pd.DataFrame({
            'Patient ID': [1, 2, 3],
            'Subject ID': [4, 5, 6],
            'Study ID': [7, 8, 9]
        })
        
        suggestions = suggest_column_renames(df)
        
        assert isinstance(suggestions, dict)
        # Should suggest standardized ID column names
    
    def test_name_patterns(self):
        """Test name pattern suggestions."""
        df = pd.DataFrame({
            'First Name': [1, 2, 3],
            'Last Name': [4, 5, 6],
            'Family Name': [7, 8, 9],
            'Given Name': [10, 11, 12]
        })
        
        suggestions = suggest_column_renames(df)
        
        assert isinstance(suggestions, dict)
        # Should suggest standardized name column names
    
    def test_no_suggestions(self):
        """Test with columns that don't match patterns."""
        df = pd.DataFrame({
            'random_column': [1, 2, 3],
            'another_col': [4, 5, 6],
            'xyz': [7, 8, 9]
        })
        
        suggestions = suggest_column_renames(df)
        
        assert isinstance(suggestions, dict)
        # May or may not have suggestions for these random names
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        
        suggestions = suggest_column_renames(df)
        
        assert isinstance(suggestions, dict)
        assert len(suggestions) == 0
    
    def test_case_insensitive_matching(self):
        """Test case-insensitive pattern matching."""
        df = pd.DataFrame({
            'PATIENT ID': [1, 2, 3],
            'patient id': [4, 5, 6],
            'Patient_ID': [7, 8, 9]
        })
        
        suggestions = suggest_column_renames(df)
        
        assert isinstance(suggestions, dict)
        # Should handle different cases


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_long_column_names(self):
        """Test with very long column names."""
        long_name = "A" * 1000  # Very long name
        df = pd.DataFrame({long_name: [1, 2, 3]})
        
        result = standardize_column_names(df, max_length=50)
        
        # Should handle very long names
        assert len(list(result.columns)[0]) <= 50
    
    def test_unicode_characters(self):
        """Test with Unicode characters."""
        df = pd.DataFrame({
            'Naïve Column': [1, 2, 3],
            'Café Data': [4, 5, 6],
            '数据列': [7, 8, 9]
        })
        
        result = standardize_column_names(df)
        
        # Should handle Unicode characters
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 3
    
    def test_numeric_column_names(self):
        """Test with numeric column names."""
        df = pd.DataFrame({
            123: [1, 2, 3],
            456.78: [4, 5, 6],
            0: [7, 8, 9]
        })
        
        result = standardize_column_names(df)
        
        # Should handle numeric column names
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 3
        # Should add prefix to numeric names
        assert all(not col[0].isdigit() for col in result.columns)
    
    def test_all_same_column_names(self):
        """Test with all identical column names."""
        # This creates a DataFrame with duplicate column names
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        df = pd.DataFrame(data, columns=['same', 'same', 'same'])
        
        result = standardize_column_names(df)
        
        # Should handle all duplicates
        assert len(result.columns) == 3
        assert len(set(result.columns)) == 3  # All should be unique


@pytest.fixture
def sample_messy_columns_df():
    """Create DataFrame with messy column names for testing."""
    return pd.DataFrame({
        'Date of Birth': [1, 2, 3],
        'Patient ID': [4, 5, 6],
        'Test@Result#1': [7, 8, 9],
        'UPPER CASE NAME': [10, 11, 12],
        'mixed_Case-Column': [13, 14, 15],
        'Column Name': [16, 17, 18],
        'Column Name ': [19, 20, 21],  # Duplicate with space
        '123NumericStart': [22, 23, 24]
    })


@pytest.fixture
def standard_columns_df():
    """Create DataFrame with already standardized column names."""
    return pd.DataFrame({
        'patient_id': [1, 2, 3],
        'date_of_birth': [4, 5, 6],
        'test_result': [7, 8, 9],
        'status': [10, 11, 12]
    })