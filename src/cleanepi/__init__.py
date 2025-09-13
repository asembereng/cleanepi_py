"""
cleanepi-python: Clean and standardize epidemiological data.

A Python package for cleaning, curating, and standardizing epidemiological data
with a focus on scalability, reliability, and security.
"""

__version__ = "0.1.0"
__author__ = "Karim Mané, Abdoelnaser Degoot"
__email__ = "karim.mane@lshtm.ac.uk"

# Core functionality imports
from .core.clean_data import clean_data
from .core.config import CleaningConfig, DateConfig, SubjectIDConfig
from .core.report import CleaningReport

# Individual cleaning functions
from .cleaning.standardize_columns import standardize_column_names
from .cleaning.standardize_dates import standardize_dates, detect_date_columns
from .cleaning.remove_duplicates import remove_duplicates
from .cleaning.remove_constants import remove_constants
from .cleaning.replace_missing import replace_missing_values
from .cleaning.convert_numeric import convert_to_numeric
from .cleaning.dictionary_cleaning import (
    clean_using_dictionary, 
    create_mapping_dictionary, 
    validate_dictionary_mappings
)
from .cleaning.validate_subject_ids import check_subject_ids, generate_subject_id_report
from .cleaning.date_sequence import (
    check_date_sequence, 
    generate_date_sequence_report, 
    detect_date_outliers
)

# Utility functions
from .utils.data_scanning import scan_data
from .utils.validation import validate_dataframe, get_memory_usage
from .utils.performance import PerformanceMonitor, performance_monitor

__all__ = [
    # Core
    "clean_data",
    "CleaningConfig",
    "DateConfig", 
    "SubjectIDConfig",
    "CleaningReport",
    # Cleaning functions
    "standardize_column_names",
    "standardize_dates",
    "detect_date_columns",
    "remove_duplicates",
    "remove_constants",
    "replace_missing_values",
    "convert_to_numeric",
    "clean_using_dictionary",
    "create_mapping_dictionary",
    "validate_dictionary_mappings", 
    "check_subject_ids",
    "generate_subject_id_report",
    "check_date_sequence",
    "generate_date_sequence_report",
    "detect_date_outliers",
    # Utils
    "scan_data",
    "validate_dataframe",
    "get_memory_usage",
    "PerformanceMonitor",
    "performance_monitor",
]