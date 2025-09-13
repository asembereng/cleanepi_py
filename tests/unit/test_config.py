"""
Comprehensive tests for the configuration module.
"""

import pytest
from datetime import date
import pandas as pd

from cleanepi.core.config import (
    DateConfig,
    SubjectIDConfig,
    MissingValueConfig,
    DuplicateConfig,
    ConstantConfig,
    NumericConfig,
    CleaningConfig,
    WebConfig
)


class TestDateConfig:
    """Test DateConfig validation and functionality."""
    
    def test_default_config(self):
        """Test default DateConfig."""
        config = DateConfig()
        assert config.target_columns is None
        assert config.formats is None
        assert config.timeframe is None
        assert config.error_tolerance == 0.4
        assert config.orders is None
    
    def test_valid_timeframe(self):
        """Test valid timeframe validation."""
        config = DateConfig(timeframe=("2020-01-01", "2023-12-31"))
        assert config.timeframe == ("2020-01-01", "2023-12-31")
    
    def test_invalid_timeframe_length(self):
        """Test invalid timeframe length."""
        with pytest.raises(ValueError):
            DateConfig(timeframe=("2020-01-01",))
        
        with pytest.raises(ValueError):
            DateConfig(timeframe=("2020-01-01", "2021-01-01", "2022-01-01"))
    
    def test_invalid_timeframe_order(self):
        """Test invalid timeframe order."""
        with pytest.raises(ValueError):
            DateConfig(timeframe=("2023-01-01", "2020-01-01"))
        
        with pytest.raises(ValueError):
            DateConfig(timeframe=("2023-01-01", "2023-01-01"))
    
    def test_invalid_timeframe_format(self):
        """Test invalid timeframe format."""
        with pytest.raises(ValueError, match="timeframe dates must be in YYYY-MM-DD format"):
            DateConfig(timeframe=("invalid-date", "2023-01-01"))
        
        with pytest.raises(ValueError, match="timeframe dates must be in YYYY-MM-DD format"):
            DateConfig(timeframe=("2023-01-01", "invalid-date"))
    
    def test_error_tolerance_bounds(self):
        """Test error tolerance bounds validation."""
        # Valid error tolerance
        config = DateConfig(error_tolerance=0.0)
        assert config.error_tolerance == 0.0
        
        config = DateConfig(error_tolerance=1.0)
        assert config.error_tolerance == 1.0
        
        config = DateConfig(error_tolerance=0.5)
        assert config.error_tolerance == 0.5
        
        # Invalid error tolerance
        with pytest.raises(ValueError):
            DateConfig(error_tolerance=-0.1)
        
        with pytest.raises(ValueError):
            DateConfig(error_tolerance=1.1)
    
    def test_all_parameters(self):
        """Test all parameters together."""
        config = DateConfig(
            target_columns=['date1', 'date2'],
            formats=['%Y-%m-%d', '%d/%m/%Y'],
            timeframe=("2020-01-01", "2023-12-31"),
            error_tolerance=0.3,
            orders={'category1': ['YMD', 'DMY']}
        )
        
        assert config.target_columns == ['date1', 'date2']
        assert config.formats == ['%Y-%m-%d', '%d/%m/%Y']
        assert config.timeframe == ("2020-01-01", "2023-12-31")
        assert config.error_tolerance == 0.3
        assert config.orders == {'category1': ['YMD', 'DMY']}


class TestSubjectIDConfig:
    """Test SubjectIDConfig validation and functionality."""
    
    def test_required_target_columns(self):
        """Test that target_columns is required."""
        config = SubjectIDConfig(target_columns=['id'])
        assert config.target_columns == ['id']
    
    def test_optional_parameters(self):
        """Test optional parameters."""
        config = SubjectIDConfig(
            target_columns=['id'],
            prefix='P',
            suffix='X',
            range=(1, 1000),
            nchar=5,
            pattern=r'^P\d{3}X$'
        )
        
        assert config.target_columns == ['id']
        assert config.prefix == 'P'
        assert config.suffix == 'X'
        assert config.range == (1, 1000)
        assert config.nchar == 5
        assert config.pattern == r'^P\d{3}X$'
    
    def test_valid_range(self):
        """Test valid range validation."""
        config = SubjectIDConfig(target_columns=['id'], range=(1, 100))
        assert config.range == (1, 100)
        
        config = SubjectIDConfig(target_columns=['id'], range=(0, 1))
        assert config.range == (0, 1)
    
    def test_invalid_range_length(self):
        """Test invalid range length."""
        with pytest.raises(ValueError):
            SubjectIDConfig(target_columns=['id'], range=(1,))
        
        with pytest.raises(ValueError):
            SubjectIDConfig(target_columns=['id'], range=(1, 2, 3))
    
    def test_invalid_range_order(self):
        """Test invalid range order."""
        with pytest.raises(ValueError):
            SubjectIDConfig(target_columns=['id'], range=(100, 1))
        
        with pytest.raises(ValueError):
            SubjectIDConfig(target_columns=['id'], range=(1, 1))
    
    def test_nchar_validation(self):
        """Test nchar validation."""
        config = SubjectIDConfig(target_columns=['id'], nchar=5)
        assert config.nchar == 5
        
        # Test minimum value
        config = SubjectIDConfig(target_columns=['id'], nchar=1)
        assert config.nchar == 1
        
        # Test invalid nchar
        with pytest.raises(ValueError):
            SubjectIDConfig(target_columns=['id'], nchar=0)
        
        with pytest.raises(ValueError):
            SubjectIDConfig(target_columns=['id'], nchar=-1)


class TestMissingValueConfig:
    """Test MissingValueConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = MissingValueConfig()
        assert config.target_columns is None
        assert "-99" in config.na_strings
        assert "N/A" in config.na_strings
        assert "NULL" in config.na_strings
        assert "" in config.na_strings
        assert "missing" in config.na_strings
        assert "unknown" in config.na_strings
        assert config.custom_na_by_column is None
    
    def test_custom_na_strings(self):
        """Test custom NA strings."""
        config = MissingValueConfig(na_strings=['custom_na', 'another_na'])
        assert config.na_strings == ['custom_na', 'another_na']
    
    def test_target_columns(self):
        """Test target columns specification."""
        config = MissingValueConfig(target_columns=['col1', 'col2'])
        assert config.target_columns == ['col1', 'col2']
    
    def test_custom_na_by_column(self):
        """Test column-specific NA strings."""
        custom_na = {
            'col1': ['na1', 'na2'],
            'col2': ['na3', 'na4']
        }
        config = MissingValueConfig(custom_na_by_column=custom_na)
        assert config.custom_na_by_column == custom_na
    
    def test_all_parameters(self):
        """Test all parameters together."""
        config = MissingValueConfig(
            target_columns=['col1', 'col2'],
            na_strings=['custom_na'],
            custom_na_by_column={'col1': ['specific_na']}
        )
        
        assert config.target_columns == ['col1', 'col2']
        assert config.na_strings == ['custom_na']
        assert config.custom_na_by_column == {'col1': ['specific_na']}


class TestDuplicateConfig:
    """Test DuplicateConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = DuplicateConfig()
        assert config.target_columns is None
        assert config.subset is None
        assert config.keep == "first"
    
    def test_target_columns(self):
        """Test target columns specification."""
        config = DuplicateConfig(target_columns=['col1', 'col2'])
        assert config.target_columns == ['col1', 'col2']
    
    def test_subset_alias(self):
        """Test subset as alias for target_columns."""
        config = DuplicateConfig(subset=['col1', 'col2'])
        assert config.subset == ['col1', 'col2']
    
    def test_keep_options(self):
        """Test keep parameter options."""
        config = DuplicateConfig(keep="first")
        assert config.keep == "first"
        
        config = DuplicateConfig(keep="last")
        assert config.keep == "last"
        
        config = DuplicateConfig(keep="False")
        assert config.keep == "False"
    
    def test_invalid_keep_option(self):
        """Test invalid keep option."""
        with pytest.raises(ValueError):
            DuplicateConfig(keep="invalid")
    
    def test_all_parameters(self):
        """Test all parameters together."""
        config = DuplicateConfig(
            target_columns=['col1'],
            subset=['col2'],
            keep="last"
        )
        
        assert config.target_columns == ['col1']
        assert config.subset == ['col2']
        assert config.keep == "last"


class TestConstantConfig:
    """Test ConstantConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ConstantConfig()
        assert config.cutoff == 1.0
        assert config.exclude_columns is None
    
    def test_cutoff_bounds(self):
        """Test cutoff bounds validation."""
        config = ConstantConfig(cutoff=0.0)
        assert config.cutoff == 0.0
        
        config = ConstantConfig(cutoff=1.0)
        assert config.cutoff == 1.0
        
        config = ConstantConfig(cutoff=0.5)
        assert config.cutoff == 0.5
        
        # Invalid cutoff values
        with pytest.raises(ValueError):
            ConstantConfig(cutoff=-0.1)
        
        with pytest.raises(ValueError):
            ConstantConfig(cutoff=1.1)
    
    def test_exclude_columns(self):
        """Test exclude columns specification."""
        config = ConstantConfig(exclude_columns=['col1', 'col2'])
        assert config.exclude_columns == ['col1', 'col2']
    
    def test_all_parameters(self):
        """Test all parameters together."""
        config = ConstantConfig(
            cutoff=0.8,
            exclude_columns=['important_col']
        )
        
        assert config.cutoff == 0.8
        assert config.exclude_columns == ['important_col']


class TestNumericConfig:
    """Test NumericConfig functionality."""
    
    def test_required_target_columns(self):
        """Test that target_columns is required."""
        config = NumericConfig(target_columns=['age', 'score'])
        assert config.target_columns == ['age', 'score']
    
    def test_default_values(self):
        """Test default values."""
        config = NumericConfig(target_columns=['age'])
        assert config.target_columns == ['age']
        assert config.lang == "en"
        assert config.errors == "coerce"
    
    def test_lang_parameter(self):
        """Test language parameter."""
        config = NumericConfig(target_columns=['age'], lang="fr")
        assert config.lang == "fr"
    
    def test_errors_options(self):
        """Test errors parameter options."""
        config = NumericConfig(target_columns=['age'], errors="raise")
        assert config.errors == "raise"
        
        config = NumericConfig(target_columns=['age'], errors="coerce")
        assert config.errors == "coerce"
        
        config = NumericConfig(target_columns=['age'], errors="ignore")
        assert config.errors == "ignore"
    
    def test_invalid_errors_option(self):
        """Test invalid errors option."""
        with pytest.raises(ValueError):
            NumericConfig(target_columns=['age'], errors="invalid")
    
    def test_all_parameters(self):
        """Test all parameters together."""
        config = NumericConfig(
            target_columns=['age', 'score'],
            lang="es",
            errors="ignore"
        )
        
        assert config.target_columns == ['age', 'score']
        assert config.lang == "es"
        assert config.errors == "ignore"


class TestCleaningConfig:
    """Test CleaningConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = CleaningConfig()
        
        assert config.standardize_column_names is True
        assert isinstance(config.replace_missing_values, MissingValueConfig)
        assert isinstance(config.remove_duplicates, DuplicateConfig)
        assert isinstance(config.remove_constants, ConstantConfig)
        assert config.standardize_dates is None
        assert config.standardize_subject_ids is None
        assert config.to_numeric is None
        assert config.dictionary is None
        assert config.check_date_sequence is None
        assert config.verbose is True
        assert config.strict_validation is False
        assert config.max_memory_usage is None
    
    def test_standardize_column_names_bool(self):
        """Test standardize_column_names as boolean."""
        config = CleaningConfig(standardize_column_names=False)
        assert config.standardize_column_names is False
        
        config = CleaningConfig(standardize_column_names=True)
        assert config.standardize_column_names is True
    
    def test_standardize_column_names_dict(self):
        """Test standardize_column_names as dict."""
        column_config = {'style': 'camelCase', 'max_length': 50}
        config = CleaningConfig(standardize_column_names=column_config)
        assert config.standardize_column_names == column_config
    
    def test_sub_configs(self):
        """Test sub-configuration objects."""
        date_config = DateConfig(target_columns=['date_col'])
        subject_id_config = SubjectIDConfig(target_columns=['id_col'])
        numeric_config = NumericConfig(target_columns=['num_col'])
        
        config = CleaningConfig(
            standardize_dates=date_config,
            standardize_subject_ids=subject_id_config,
            to_numeric=numeric_config
        )
        
        assert config.standardize_dates == date_config
        assert config.standardize_subject_ids == subject_id_config
        assert config.to_numeric == numeric_config
    
    def test_dictionary_config(self):
        """Test dictionary configuration."""
        dictionary = {
            'status': {'pos': 'positive', 'neg': 'negative'},
            'gender': {'M': 'Male', 'F': 'Female'}
        }
        
        config = CleaningConfig(dictionary=dictionary)
        assert config.dictionary == dictionary
    
    def test_check_date_sequence(self):
        """Test date sequence checking configuration."""
        date_columns = ['visit_date', 'follow_up_date']
        config = CleaningConfig(check_date_sequence=date_columns)
        assert config.check_date_sequence == date_columns
    
    def test_global_settings(self):
        """Test global settings."""
        config = CleaningConfig(
            verbose=False,
            strict_validation=True,
            max_memory_usage="1GB"
        )
        
        assert config.verbose is False
        assert config.strict_validation is True
        assert config.max_memory_usage == "1GB"
    
    def test_forbid_extra_fields(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValueError):
            CleaningConfig(unknown_field="value")
    
    def test_comprehensive_config(self):
        """Test comprehensive configuration with all options."""
        config = CleaningConfig(
            standardize_column_names={'style': 'snake_case'},
            replace_missing_values=MissingValueConfig(na_strings=['NA']),
            remove_duplicates=DuplicateConfig(keep='last'),
            remove_constants=ConstantConfig(cutoff=0.95),
            standardize_dates=DateConfig(target_columns=['date_col']),
            standardize_subject_ids=SubjectIDConfig(target_columns=['id_col']),
            to_numeric=NumericConfig(target_columns=['num_col']),
            dictionary={'col': {'old': 'new'}},
            check_date_sequence=['date1', 'date2'],
            verbose=False,
            strict_validation=True,
            max_memory_usage="500MB"
        )
        
        assert isinstance(config.standardize_column_names, dict)
        assert isinstance(config.replace_missing_values, MissingValueConfig)
        assert isinstance(config.remove_duplicates, DuplicateConfig)
        assert isinstance(config.remove_constants, ConstantConfig)
        assert isinstance(config.standardize_dates, DateConfig)
        assert isinstance(config.standardize_subject_ids, SubjectIDConfig)
        assert isinstance(config.to_numeric, NumericConfig)
        assert config.dictionary == {'col': {'old': 'new'}}
        assert config.check_date_sequence == ['date1', 'date2']
        assert config.verbose is False
        assert config.strict_validation is True
        assert config.max_memory_usage == "500MB"


class TestWebConfig:
    """Test WebConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = WebConfig()
        
        assert config.max_file_size == 100 * 1024 * 1024  # 100MB
        assert ".csv" in config.allowed_file_types
        assert ".xlsx" in config.allowed_file_types
        assert ".parquet" in config.allowed_file_types
        assert ".json" in config.allowed_file_types
        assert config.temp_dir == "/tmp/cleanepi"
        assert config.enable_async is True
        assert config.chunk_size == 10000
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = WebConfig(
            max_file_size=50 * 1024 * 1024,  # 50MB
            allowed_file_types=[".csv", ".xlsx"],
            temp_dir="/custom/temp",
            enable_async=False,
            chunk_size=5000
        )
        
        assert config.max_file_size == 50 * 1024 * 1024
        assert config.allowed_file_types == [".csv", ".xlsx"]
        assert config.temp_dir == "/custom/temp"
        assert config.enable_async is False
        assert config.chunk_size == 5000
    
    def test_forbid_extra_fields(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValueError):
            WebConfig(unknown_field="value")


@pytest.fixture
def sample_date_config():
    """Sample DateConfig for testing."""
    return DateConfig(
        target_columns=['date_col'],
        timeframe=("2020-01-01", "2023-12-31"),
        error_tolerance=0.3
    )


@pytest.fixture
def sample_cleaning_config():
    """Sample CleaningConfig for testing."""
    return CleaningConfig(
        verbose=True,
        strict_validation=False,
        replace_missing_values=MissingValueConfig(na_strings=['NA', 'NULL'])
    )