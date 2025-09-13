"""
Comprehensive tests for the validation utilities module.
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open

from cleanepi.utils.validation import (
    validate_dataframe,
    validate_config,
    validate_columns_exist,
    validate_column_types,
    check_memory_usage,
    _parse_memory_string,
    sanitize_column_names,
    validate_file_safety,
    detect_encoding
)
from cleanepi.core.config import CleaningConfig, SubjectIDConfig, NumericConfig


class TestValidateDataFrame:
    """Test DataFrame validation."""
    
    def test_valid_dataframe(self):
        """Test with valid DataFrame."""
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        # Should not raise any exception
        validate_dataframe(df)
    
    def test_non_dataframe_input(self):
        """Test with non-DataFrame input."""
        with pytest.raises(TypeError, match="Expected pandas DataFrame"):
            validate_dataframe("not a dataframe")
        
        with pytest.raises(TypeError, match="Expected pandas DataFrame"):
            validate_dataframe([1, 2, 3])
        
        with pytest.raises(TypeError, match="Expected pandas DataFrame"):
            validate_dataframe({'a': [1, 2, 3]})
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame is empty"):
            validate_dataframe(df)
    
    def test_minimum_rows_validation(self):
        """Test minimum rows validation."""
        df = pd.DataFrame({'col1': [1], 'col2': ['a']})
        
        # Should pass with default min_rows=1
        validate_dataframe(df)
        
        # Should fail with min_rows=2
        with pytest.raises(ValueError, match="DataFrame has 1 rows, minimum 2 required"):
            validate_dataframe(df, min_rows=2)
    
    def test_minimum_columns_validation(self):
        """Test minimum columns validation."""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        
        # Should pass with default min_cols=1
        validate_dataframe(df)
        
        # Should fail with min_cols=2
        with pytest.raises(ValueError, match="DataFrame has 1 columns, minimum 2 required"):
            validate_dataframe(df, min_cols=2)


class TestValidateConfig:
    """Test configuration validation."""
    
    def test_valid_config(self):
        """Test with valid configuration."""
        config = CleaningConfig()
        # Should not raise any exception
        validate_config(config)
    
    def test_non_config_input(self):
        """Test with non-config input."""
        with pytest.raises(TypeError, match="Expected CleaningConfig"):
            validate_config("not a config")
        
        with pytest.raises(TypeError, match="Expected CleaningConfig"):
            validate_config({'verbose': True})
    
    def test_subject_id_config_without_target_columns(self):
        """Test subject ID config validation."""
        subject_id_config = SubjectIDConfig(target_columns=[])
        config = CleaningConfig(standardize_subject_ids=subject_id_config)
        
        with pytest.raises(ValueError, match="standardize_subject_ids requires target_columns"):
            validate_config(config)
    
    def test_numeric_config_without_target_columns(self):
        """Test numeric config validation."""
        numeric_config = NumericConfig(target_columns=[])
        config = CleaningConfig(to_numeric=numeric_config)
        
        with pytest.raises(ValueError, match="to_numeric requires target_columns"):
            validate_config(config)
    
    def test_valid_subject_id_config(self):
        """Test valid subject ID config."""
        subject_id_config = SubjectIDConfig(target_columns=['id'])
        config = CleaningConfig(standardize_subject_ids=subject_id_config)
        
        # Should not raise any exception
        validate_config(config)
    
    def test_valid_numeric_config(self):
        """Test valid numeric config."""
        numeric_config = NumericConfig(target_columns=['age'])
        config = CleaningConfig(to_numeric=numeric_config)
        
        # Should not raise any exception
        validate_config(config)


class TestValidateColumnsExist:
    """Test column existence validation."""
    
    def test_existing_columns(self):
        """Test with existing columns."""
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        
        # Should not raise any exception
        validate_columns_exist(df, ['col1'])
        validate_columns_exist(df, ['col1', 'col2'])
        validate_columns_exist(df, [])  # Empty list should work
    
    def test_missing_columns(self):
        """Test with missing columns."""
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        
        with pytest.raises(ValueError, match="Columns \\['col3'\\] not found"):
            validate_columns_exist(df, ['col3'])
        
        with pytest.raises(ValueError, match="Columns \\['col3', 'col4'\\] not found"):
            validate_columns_exist(df, ['col1', 'col3', 'col4'])
    
    def test_with_operation_name(self):
        """Test error message with operation name."""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="test_operation: Columns"):
            validate_columns_exist(df, ['missing'], operation="test_operation")


class TestValidateColumnTypes:
    """Test column type validation."""
    
    def test_valid_column_types(self):
        """Test with valid column types."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.0, 2.0, 3.0],
            'str_col': ['a', 'b', 'c']
        })
        
        # Should not raise any exception
        validate_column_types(df, ['int_col'], [int, 'int64'])
        validate_column_types(df, ['str_col'], [object])
    
    def test_invalid_column_types(self):
        """Test with invalid column types."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'str_col': ['a', 'b', 'c']
        })
        
        with pytest.raises(ValueError, match="Column 'str_col' has type"):
            validate_column_types(df, ['str_col'], [int])
    
    def test_missing_columns_in_type_validation(self):
        """Test type validation with missing columns."""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Columns \\['missing'\\] not found"):
            validate_column_types(df, ['missing'], [int])


class TestCheckMemoryUsage:
    """Test memory usage checking."""
    
    def test_no_limit_specified(self):
        """Test with no memory limit."""
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        
        # Should not raise any exception
        check_memory_usage(df)
        check_memory_usage(df, max_memory=None)
    
    def test_within_memory_limit(self):
        """Test DataFrame within memory limit."""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        
        # Should not raise any exception with generous limit
        check_memory_usage(df, max_memory="1GB")
        check_memory_usage(df, max_memory="100MB")
    
    def test_exceeds_memory_limit(self):
        """Test DataFrame exceeding memory limit."""
        # Create a larger DataFrame
        df = pd.DataFrame({'col1': range(10000), 'col2': ['text'] * 10000})
        
        with pytest.raises(MemoryError, match="DataFrame memory usage.*exceeds limit"):
            check_memory_usage(df, max_memory="1KB")


class TestParseMemoryString:
    """Test memory string parsing."""
    
    def test_gigabytes(self):
        """Test gigabyte parsing."""
        assert _parse_memory_string("1GB") == 1024**3
        assert _parse_memory_string("2gb") == 2 * 1024**3
        assert _parse_memory_string("1.5GB") == int(1.5 * 1024**3)
        assert _parse_memory_string(" 1GB ") == 1024**3
    
    def test_megabytes(self):
        """Test megabyte parsing."""
        assert _parse_memory_string("1MB") == 1024**2
        assert _parse_memory_string("500mb") == 500 * 1024**2
        assert _parse_memory_string("1.5MB") == int(1.5 * 1024**2)
    
    def test_kilobytes(self):
        """Test kilobyte parsing."""
        assert _parse_memory_string("1KB") == 1024
        assert _parse_memory_string("500kb") == 500 * 1024
        assert _parse_memory_string("1.5KB") == int(1.5 * 1024)
    
    def test_bytes(self):
        """Test byte parsing."""
        assert _parse_memory_string("1000B") == 1000
        assert _parse_memory_string("1000b") == 1000
        assert _parse_memory_string("1000") == 1000  # No suffix
    
    def test_invalid_memory_string(self):
        """Test invalid memory strings."""
        with pytest.raises(ValueError):
            _parse_memory_string("invalid")


class TestSanitizeColumnNames:
    """Test column name sanitization."""
    
    def test_normal_columns(self):
        """Test with normal column names."""
        columns = ['col1', 'col_2', 'column-3', 'Column.4']
        result = sanitize_column_names(columns)
        assert result == ['col1', 'col_2', 'column-3', 'Column.4']
    
    def test_dangerous_characters(self):
        """Test with potentially dangerous characters."""
        columns = ['col;1', 'col<script>', 'col"with"quotes', 'col with spaces']
        result = sanitize_column_names(columns)
        expected = ['col_1', 'col_script_', 'col_with_quotes', 'col_with_spaces']
        assert result == expected
    
    def test_special_characters(self):
        """Test with various special characters."""
        columns = ['col@#$%', 'col[]*&', 'col(){}/\\']
        result = sanitize_column_names(columns)
        expected = ['col____', 'col____', 'col______']
        assert result == expected
    
    def test_empty_and_none(self):
        """Test with empty or None values."""
        columns = ['', None, 'valid_col']
        result = sanitize_column_names(columns)
        assert result == ['', 'None', 'valid_col']


class TestValidateFileSafety:
    """Test file safety validation."""
    
    def test_safe_file_paths(self):
        """Test with safe file paths."""
        # These should not raise exceptions
        validate_file_safety("test.csv")
        validate_file_safety("data/test.xlsx")
        validate_file_safety("/tmp/test.parquet")
    
    def test_path_traversal_attack(self):
        """Test path traversal detection."""
        dangerous_paths = [
            "../../../etc/passwd",
            "data/../../../etc/passwd", 
            "test/../../../../../etc/passwd"
        ]
        
        for path in dangerous_paths:
            with pytest.raises(ValueError, match="Path traversal detected"):
                validate_file_safety(path)
    
    def test_allowed_extensions(self):
        """Test file extension validation."""
        # Should pass with allowed extensions
        validate_file_safety("test.csv", allowed_extensions=[".csv", ".xlsx"])
        validate_file_safety("test.xlsx", allowed_extensions=[".csv", ".xlsx"])
        
        # Should fail with disallowed extensions
        with pytest.raises(ValueError, match="File extension .txt not allowed"):
            validate_file_safety("test.txt", allowed_extensions=[".csv", ".xlsx"])
    
    def test_case_insensitive_extensions(self):
        """Test case-insensitive extension checking."""
        validate_file_safety("test.CSV", allowed_extensions=[".csv"])
        validate_file_safety("test.xlsx", allowed_extensions=[".XLSX"])
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.stat')
    def test_file_size_limit(self, mock_stat, mock_exists):
        """Test file size checking."""
        mock_exists.return_value = True
        
        # Mock file stats for large file
        mock_stat.return_value.st_size = 600 * 1024 * 1024  # 600MB
        
        with pytest.raises(ValueError, match="File size.*exceeds limit"):
            validate_file_safety("large_file.csv")
    
    @patch('pathlib.Path.exists')
    def test_nonexistent_file(self, mock_exists):
        """Test with non-existent file."""
        mock_exists.return_value = False
        
        # Should not raise exception for non-existent files
        validate_file_safety("nonexistent.csv")


class TestDetectEncoding:
    """Test encoding detection."""
    
    def test_detect_encoding_utf8(self):
        """Test encoding detection for UTF-8 file."""
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
            f.write("test,data\n1,hello\n2,world\n")
            temp_path = f.name
        
        try:
            with patch('chardet.detect') as mock_detect:
                mock_detect.return_value = {'encoding': 'utf-8'}
                encoding = detect_encoding(temp_path)
                assert encoding == 'utf-8'
        finally:
            os.unlink(temp_path)
    
    def test_detect_encoding_latin1(self):
        """Test encoding detection for Latin-1 file."""
        with tempfile.NamedTemporaryFile(mode='w', encoding='latin-1', delete=False) as f:
            f.write("test,data\n1,café\n2,naïve\n")
            temp_path = f.name
        
        try:
            with patch('chardet.detect') as mock_detect:
                mock_detect.return_value = {'encoding': 'latin-1'}
                encoding = detect_encoding(temp_path)
                assert encoding == 'latin-1'
        finally:
            os.unlink(temp_path)
    
    def test_detect_encoding_fallback(self):
        """Test encoding detection fallback."""
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
            f.write("test,data\n")
            temp_path = f.name
        
        try:
            with patch('chardet.detect') as mock_detect:
                mock_detect.return_value = {}  # No encoding detected
                encoding = detect_encoding(temp_path)
                assert encoding == 'utf-8'  # Fallback
        finally:
            os.unlink(temp_path)
    
    def test_detect_encoding_sample_size(self):
        """Test encoding detection with custom sample size."""
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
            f.write("test,data\n" * 100)
            temp_path = f.name
        
        try:
            with patch('chardet.detect') as mock_detect:
                mock_detect.return_value = {'encoding': 'utf-8'}
                
                detect_encoding(temp_path, sample_size=500)
                
                # Verify chardet.detect was called
                mock_detect.assert_called_once()
        finally:
            os.unlink(temp_path)
    
    @patch('cleanepi.utils.validation.validate_file_safety')
    def test_detect_encoding_validates_file_safety(self, mock_validate):
        """Test that detect_encoding validates file safety."""
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
            f.write("test")
            temp_path = f.name
        
        try:
            with patch('chardet.detect') as mock_detect:
                mock_detect.return_value = {'encoding': 'utf-8'}
                
                detect_encoding(temp_path)
                
                # Verify file safety validation was called
                mock_validate.assert_called_once_with(temp_path)
        finally:
            os.unlink(temp_path)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'salary': [50000.0, 60000.0, 55000.0, 52000.0, 58000.0]
    })


@pytest.fixture
def sample_config():
    """Create a sample cleaning configuration."""
    return CleaningConfig(
        verbose=True,
        strict_validation=False
    )