"""
Comprehensive tests for the CLI module.
"""

import pytest
import sys
import tempfile
import os
import json
import pandas as pd
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
from io import StringIO

from cleanepi.cli import (
    setup_logging,
    load_data,
    save_data,
    create_config_from_args,
    main
)
from cleanepi.core.config import CleaningConfig, MissingValueConfig, DuplicateConfig, ConstantConfig


class TestSetupLogging:
    """Test setup_logging functionality."""
    
    @patch('cleanepi.cli.logger')
    def test_setup_logging_verbose(self, mock_logger):
        """Test logging setup in verbose mode."""
        setup_logging(verbose=True)
        
        # Should remove existing handlers and add new one
        mock_logger.remove.assert_called_once()
        mock_logger.add.assert_called_once()
        
        # Check that DEBUG level is used for verbose
        call_args = mock_logger.add.call_args
        assert call_args[1]['level'] == "DEBUG"
    
    @patch('cleanepi.cli.logger')
    def test_setup_logging_normal(self, mock_logger):
        """Test logging setup in normal mode."""
        setup_logging(verbose=False)
        
        # Should remove existing handlers and add new one
        mock_logger.remove.assert_called_once()
        mock_logger.add.assert_called_once()
        
        # Check that INFO level is used for normal
        call_args = mock_logger.add.call_args
        assert call_args[1]['level'] == "INFO"


class TestLoadData:
    """Test load_data functionality."""
    
    def test_load_csv_file(self):
        """Test loading CSV file."""
        csv_content = "id,name,age\n1,Alice,25\n2,Bob,30\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            with patch('cleanepi.cli.detect_encoding', return_value='utf-8'):
                result = load_data(temp_path)
                
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert list(result.columns) == ['id', 'name', 'age']
        finally:
            os.unlink(temp_path)
    
    def test_load_excel_file(self):
        """Test loading Excel file."""
        df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            temp_path = f.name
            df.to_excel(temp_path, index=False)
        
        try:
            result = load_data(temp_path)
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert 'id' in result.columns
            assert 'name' in result.columns
        finally:
            os.unlink(temp_path)
    
    def test_load_parquet_file(self):
        """Test loading Parquet file."""
        df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_path = f.name
            df.to_parquet(temp_path, index=False)
        
        try:
            result = load_data(temp_path)
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert 'id' in result.columns
            assert 'name' in result.columns
        finally:
            os.unlink(temp_path)
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_data("nonexistent_file.csv")
    
    def test_load_unsupported_format(self):
        """Test loading unsupported file format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name
            f.write("some text content")
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                load_data(temp_path)
        finally:
            os.unlink(temp_path)
    
    @patch('cleanepi.cli.validate_file_safety')
    def test_load_data_validates_safety(self, mock_validate):
        """Test that load_data validates file safety."""
        csv_content = "id,name\n1,Alice\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            with patch('cleanepi.cli.detect_encoding', return_value='utf-8'):
                load_data(temp_path)
                
            # Should call validate_file_safety
            mock_validate.assert_called_once()
        finally:
            os.unlink(temp_path)


class TestSaveData:
    """Test save_data functionality."""
    
    def test_save_csv_file(self):
        """Test saving CSV file."""
        df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            save_data(df, temp_path)
            
            # Verify file was created and contains correct data
            assert os.path.exists(temp_path)
            loaded_df = pd.read_csv(temp_path)
            assert len(loaded_df) == 2
            assert list(loaded_df.columns) == ['id', 'name']
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_excel_file(self):
        """Test saving Excel file."""
        df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            temp_path = f.name
        
        try:
            save_data(df, temp_path)
            
            # Verify file was created and contains correct data
            assert os.path.exists(temp_path)
            loaded_df = pd.read_excel(temp_path)
            assert len(loaded_df) == 2
            assert 'id' in loaded_df.columns
            assert 'name' in loaded_df.columns
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_parquet_file(self):
        """Test saving Parquet file."""
        df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_path = f.name
        
        try:
            save_data(df, temp_path)
            
            # Verify file was created and contains correct data
            assert os.path.exists(temp_path)
            loaded_df = pd.read_parquet(temp_path)
            assert len(loaded_df) == 2
            assert 'id' in loaded_df.columns
            assert 'name' in loaded_df.columns
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_unsupported_format(self):
        """Test saving unsupported file format."""
        df = pd.DataFrame({'id': [1, 2]})
        
        with pytest.raises(ValueError, match="Unsupported output format"):
            save_data(df, "output.txt")
    
    def test_save_creates_directory(self):
        """Test that save_data creates parent directories."""
        df = pd.DataFrame({'id': [1, 2]})
        
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, "subdir", "data.csv")
            
            save_data(df, nested_path)
            
            # Verify file was created in nested directory
            assert os.path.exists(nested_path)
            loaded_df = pd.read_csv(nested_path)
            assert len(loaded_df) == 2


class TestCreateConfigFromArgs:
    """Test create_config_from_args functionality."""
    
    def test_basic_config_creation(self):
        """Test basic config creation."""
        args = MagicMock()
        args.na_strings = None
        args.replace_missing = False
        args.remove_duplicates = False
        args.remove_constants = False
        args.standardize_columns = True
        args.verbose = True
        
        config = create_config_from_args(args)
        
        assert isinstance(config, CleaningConfig)
        assert config.standardize_column_names is True
        assert config.verbose is True
        assert config.replace_missing_values is None
        assert config.remove_duplicates is None
        assert config.remove_constants is None
    
    def test_missing_value_config(self):
        """Test missing value configuration."""
        args = MagicMock()
        args.na_strings = "NA,NULL,missing"
        args.replace_missing = True
        args.remove_duplicates = False
        args.remove_constants = False
        args.standardize_columns = False
        args.verbose = False
        
        config = create_config_from_args(args)
        
        assert isinstance(config.replace_missing_values, MissingValueConfig)
        assert config.replace_missing_values.na_strings == ['NA', 'NULL', 'missing']
    
    def test_duplicate_config(self):
        """Test duplicate removal configuration."""
        args = MagicMock()
        args.na_strings = None
        args.replace_missing = False
        args.remove_duplicates = True
        args.duplicate_keep = "last"
        args.remove_constants = False
        args.standardize_columns = False
        args.verbose = False
        
        config = create_config_from_args(args)
        
        assert isinstance(config.remove_duplicates, DuplicateConfig)
        assert config.remove_duplicates.keep == "last"
    
    def test_constant_config(self):
        """Test constant removal configuration."""
        args = MagicMock()
        args.na_strings = None
        args.replace_missing = False
        args.remove_duplicates = False
        args.remove_constants = True
        args.constant_cutoff = 0.95
        args.standardize_columns = False
        args.verbose = False
        
        config = create_config_from_args(args)
        
        assert isinstance(config.remove_constants, ConstantConfig)
        assert config.remove_constants.cutoff == 0.95
    
    def test_all_options_config(self):
        """Test configuration with all options enabled."""
        args = MagicMock()
        args.na_strings = "NA,NULL"
        args.replace_missing = True
        args.remove_duplicates = True
        args.duplicate_keep = "first"
        args.remove_constants = True
        args.constant_cutoff = 1.0
        args.standardize_columns = True
        args.verbose = True
        
        config = create_config_from_args(args)
        
        assert config.standardize_column_names is True
        assert isinstance(config.replace_missing_values, MissingValueConfig)
        assert isinstance(config.remove_duplicates, DuplicateConfig)
        assert isinstance(config.remove_constants, ConstantConfig)
        assert config.verbose is True


class TestMainFunction:
    """Test main CLI function."""
    
    @patch('cleanepi.cli.clean_data')
    @patch('cleanepi.cli.save_data')
    @patch('cleanepi.cli.load_data')
    @patch('cleanepi.cli.setup_logging')
    @patch('sys.argv', ['cleanepi', 'input.csv'])
    def test_basic_execution(self, mock_setup_logging, mock_load, mock_save, mock_clean):
        """Test basic CLI execution."""
        # Mock data and report
        mock_data = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        mock_cleaned = pd.DataFrame({'id': [1, 2], 'name': ['alice', 'bob']})
        mock_report = MagicMock()
        mock_report.total_rows_removed = 0
        mock_report.total_columns_removed = 1
        
        mock_load.return_value = mock_data
        mock_clean.return_value = (mock_cleaned, mock_report)
        
        with patch('builtins.print') as mock_print:
            main()
        
        # Verify functions were called
        mock_setup_logging.assert_called_once()
        mock_load.assert_called_once_with('input.csv')
        mock_clean.assert_called_once()
        mock_save.assert_called_once()
        mock_print.assert_called()
    
    @patch('cleanepi.cli.clean_data')
    @patch('cleanepi.cli.save_data')
    @patch('cleanepi.cli.load_data')
    @patch('cleanepi.cli.setup_logging')
    @patch('sys.argv', ['cleanepi', 'input.csv', '-o', 'output.csv'])
    def test_with_output_file(self, mock_setup_logging, mock_load, mock_save, mock_clean):
        """Test CLI with specified output file."""
        mock_data = pd.DataFrame({'id': [1, 2]})
        mock_cleaned = pd.DataFrame({'id': [1, 2]})
        mock_report = MagicMock()
        mock_report.total_rows_removed = 0
        mock_report.total_columns_removed = 0
        
        mock_load.return_value = mock_data
        mock_clean.return_value = (mock_cleaned, mock_report)
        
        main()
        
        # Should save to specified output file
        mock_save.assert_called_once()
        save_call_args = mock_save.call_args
        assert save_call_args[0][1] == 'output.csv'
    
    @patch('cleanepi.cli.clean_data')
    @patch('cleanepi.cli.save_data')
    @patch('cleanepi.cli.load_data')
    @patch('cleanepi.cli.setup_logging')
    @patch('sys.argv', ['cleanepi', 'input.csv', '--config', 'config.json'])
    def test_with_config_file(self, mock_setup_logging, mock_load, mock_save, mock_clean):
        """Test CLI with configuration file."""
        config_dict = {
            "standardize_column_names": True,
            "verbose": True
        }
        
        mock_data = pd.DataFrame({'id': [1, 2]})
        mock_cleaned = pd.DataFrame({'id': [1, 2]})
        mock_report = MagicMock()
        mock_report.total_rows_removed = 0
        mock_report.total_columns_removed = 0
        
        mock_load.return_value = mock_data
        mock_clean.return_value = (mock_cleaned, mock_report)
        
        with patch('builtins.open', mock_open(read_data=json.dumps(config_dict))):
            main()
        
        # Should load config from file
        mock_clean.assert_called_once()
        config_used = mock_clean.call_args[0][1]
        assert isinstance(config_used, CleaningConfig)
    
    @patch('cleanepi.cli.clean_data')
    @patch('cleanepi.cli.save_data')
    @patch('cleanepi.cli.load_data')
    @patch('cleanepi.cli.setup_logging')
    @patch('sys.argv', ['cleanepi', 'input.csv', '--report', 'report.json'])
    def test_with_report_output(self, mock_setup_logging, mock_load, mock_save, mock_clean):
        """Test CLI with report output."""
        mock_data = pd.DataFrame({'id': [1, 2]})
        mock_cleaned = pd.DataFrame({'id': [1, 2]})
        mock_report = MagicMock()
        mock_report.total_rows_removed = 0
        mock_report.total_columns_removed = 0
        
        mock_load.return_value = mock_data
        mock_clean.return_value = (mock_cleaned, mock_report)
        
        main()
        
        # Should save report
        mock_report.to_file.assert_called_once_with('report.json')
    
    @patch('cleanepi.cli.clean_data')
    @patch('cleanepi.cli.save_data')
    @patch('cleanepi.cli.load_data')
    @patch('cleanepi.cli.setup_logging')
    @patch('sys.argv', ['cleanepi', 'input.csv', '--preview', '5'])
    def test_with_preview(self, mock_setup_logging, mock_load, mock_save, mock_clean):
        """Test CLI with data preview."""
        mock_data = pd.DataFrame({'id': [1, 2, 3, 4, 5, 6]})
        mock_cleaned = pd.DataFrame({'id': [1, 2, 3, 4, 5, 6]})
        mock_report = MagicMock()
        mock_report.total_rows_removed = 0
        mock_report.total_columns_removed = 0
        
        mock_load.return_value = mock_data
        mock_clean.return_value = (mock_cleaned, mock_report)
        
        with patch('builtins.print') as mock_print:
            main()
        
        # Should print preview
        mock_print.assert_called()
        # Check if preview-related content was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any('rows of cleaned data' in call for call in print_calls)
    
    @patch('cleanepi.cli.clean_data')
    @patch('cleanepi.cli.save_data')
    @patch('cleanepi.cli.load_data')
    @patch('cleanepi.cli.setup_logging')
    @patch('sys.argv', ['cleanepi', 'input.csv', '--standardize-columns', '--verbose'])
    def test_with_cleaning_options(self, mock_setup_logging, mock_load, mock_save, mock_clean):
        """Test CLI with various cleaning options."""
        mock_data = pd.DataFrame({'id': [1, 2]})
        mock_cleaned = pd.DataFrame({'id': [1, 2]})
        mock_report = MagicMock()
        mock_report.total_rows_removed = 0
        mock_report.total_columns_removed = 0
        
        mock_load.return_value = mock_data
        mock_clean.return_value = (mock_cleaned, mock_report)
        
        main()
        
        # Should create config with specified options
        mock_clean.assert_called_once()
        config_used = mock_clean.call_args[0][1]
        assert config_used.standardize_column_names is True
        assert config_used.verbose is True
    
    @patch('cleanepi.cli.load_data')
    @patch('cleanepi.cli.setup_logging')
    @patch('sys.argv', ['cleanepi', 'nonexistent.csv'])
    def test_file_not_found_error(self, mock_setup_logging, mock_load):
        """Test CLI with file not found error."""
        mock_load.side_effect = FileNotFoundError("File not found")
        
        with pytest.raises(SystemExit) as exc_info:
            with patch('cleanepi.cli.logger') as mock_logger:
                main()
        
        # Should exit with code 1
        assert exc_info.value.code == 1
    
    @patch('cleanepi.cli.clean_data')
    @patch('cleanepi.cli.load_data')
    @patch('cleanepi.cli.setup_logging')
    @patch('sys.argv', ['cleanepi', 'input.csv'])
    def test_cleaning_error(self, mock_setup_logging, mock_load, mock_clean):
        """Test CLI with cleaning error."""
        mock_data = pd.DataFrame({'id': [1, 2]})
        mock_load.return_value = mock_data
        mock_clean.side_effect = ValueError("Cleaning failed")
        
        with pytest.raises(SystemExit) as exc_info:
            with patch('cleanepi.cli.logger') as mock_logger:
                main()
        
        # Should exit with code 1
        assert exc_info.value.code == 1
    
    @patch('sys.argv', ['cleanepi', '--version'])
    def test_version_option(self):
        """Test --version option."""
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        # Should exit with code 0 for version
        assert exc_info.value.code == 0
    
    @patch('cleanepi.cli.clean_data')
    @patch('cleanepi.cli.save_data')
    @patch('cleanepi.cli.load_data')
    @patch('cleanepi.cli.setup_logging')
    @patch('sys.argv', ['cleanepi', 'input.csv', '--remove-duplicates', '--duplicate-keep', 'last'])
    def test_duplicate_options(self, mock_setup_logging, mock_load, mock_save, mock_clean):
        """Test CLI with duplicate removal options."""
        mock_data = pd.DataFrame({'id': [1, 2]})
        mock_cleaned = pd.DataFrame({'id': [1, 2]})
        mock_report = MagicMock()
        mock_report.total_rows_removed = 0
        mock_report.total_columns_removed = 0
        
        mock_load.return_value = mock_data
        mock_clean.return_value = (mock_cleaned, mock_report)
        
        main()
        
        # Should create config with duplicate options
        mock_clean.assert_called_once()
        config_used = mock_clean.call_args[0][1]
        assert isinstance(config_used.remove_duplicates, DuplicateConfig)
        assert config_used.remove_duplicates.keep == "last"
    
    @patch('cleanepi.cli.clean_data')
    @patch('cleanepi.cli.save_data')
    @patch('cleanepi.cli.load_data')
    @patch('cleanepi.cli.setup_logging')
    @patch('sys.argv', ['cleanepi', 'input.csv', '--replace-missing', '--na-strings', 'NA,NULL,missing'])
    def test_missing_value_options(self, mock_setup_logging, mock_load, mock_save, mock_clean):
        """Test CLI with missing value options."""
        mock_data = pd.DataFrame({'id': [1, 2]})
        mock_cleaned = pd.DataFrame({'id': [1, 2]})
        mock_report = MagicMock()
        mock_report.total_rows_removed = 0
        mock_report.total_columns_removed = 0
        
        mock_load.return_value = mock_data
        mock_clean.return_value = (mock_cleaned, mock_report)
        
        main()
        
        # Should create config with missing value options
        mock_clean.assert_called_once()
        config_used = mock_clean.call_args[0][1]
        assert isinstance(config_used.replace_missing_values, MissingValueConfig)
        assert config_used.replace_missing_values.na_strings == ['NA', 'NULL', 'missing']


@pytest.fixture
def sample_csv_file():
    """Create sample CSV file for testing."""
    csv_content = "id,name,age\n1,Alice,25\n2,Bob,30\n3,Charlie,35\n"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        return f.name


@pytest.fixture
def sample_config_file():
    """Create sample config file for testing."""
    config = {
        "standardize_column_names": True,
        "verbose": True,
        "replace_missing_values": {
            "na_strings": ["NA", "NULL", "missing"]
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        return f.name