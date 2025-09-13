"""
cleanepi-python: Clean and standardize epidemiological data.

A Python package for cleaning, curating, and standardizing epidemiological data
with a focus on scalability, reliability, and security.
"""

__version__ = "0.1.0"
__author__ = "Karim Mané, Abdoelnaser Degoot"
__email__ = "karim.mane@lshtm.ac.uk"

from .cleaning.convert_numeric import convert_to_numeric
from .cleaning.date_sequence import check_date_sequence
from .cleaning.dictionary_cleaning import clean_using_dictionary
from .cleaning.remove_constants import remove_constants
from .cleaning.remove_duplicates import remove_duplicates
from .cleaning.replace_missing import replace_missing_values

# Individual cleaning functions
from .cleaning.standardize_columns import standardize_column_names
from .cleaning.standardize_dates import standardize_dates
from .cleaning.validate_subject_ids import check_subject_ids

# Core functionality imports
from .core.clean_data import clean_data
from .core.config import CleaningConfig, DateConfig, SubjectIDConfig
from .core.report import CleaningReport

# Utility functions
from .utils.data_scanning import scan_data
from .utils.validation import validate_dataframe

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
    "remove_duplicates",
    "remove_constants",
    "replace_missing_values",
    "convert_to_numeric",
    "clean_using_dictionary",
    "check_subject_ids",
    "check_date_sequence",
    # Utils
    "scan_data",
    "validate_dataframe",
]
