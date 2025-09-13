"""Utility functions for cleanepi package."""

import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict, Any


def get_target_column_names(data: pd.DataFrame, 
                          target_columns: Optional[List[str]] = None,
                          cols: Optional[List[str]] = None) -> List[str]:
    """
    Get target column names from the dataframe.
    
    Args:
        data: Input DataFrame
        target_columns: List of target column names, if None use all columns
        cols: Additional columns to consider
        
    Returns:
        List of column names to process
    """
    if target_columns is None:
        return list(data.columns)
    else:
        return [col for col in target_columns if col in data.columns]


def add_to_report(data: pd.DataFrame, 
                 key: str, 
                 value: Any) -> pd.DataFrame:
    """
    Add information to the cleaning report.
    
    Args:
        data: Input DataFrame
        key: Report key
        value: Report value
        
    Returns:
        DataFrame with report information added as attribute
    """
    if not hasattr(data, '_cleanepi_report'):
        data._cleanepi_report = {}
    data._cleanepi_report[key] = value
    return data


def get_report(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Get the cleaning report from a DataFrame.
    
    Args:
        data: DataFrame with cleaning report
        
    Returns:
        Dictionary containing the cleaning report
    """
    return getattr(data, '_cleanepi_report', {})


def retrieve_column_names(data: pd.DataFrame, 
                         target_columns: Optional[List[str]]) -> Optional[List[str]]:
    """
    Retrieve and validate column names from the dataframe.
    
    Args:
        data: Input DataFrame
        target_columns: List of target column names
        
    Returns:
        Validated list of column names
    """
    if target_columns is None:
        return None
    
    # Check if all target columns exist in the dataframe
    missing_cols = [col for col in target_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in data: {missing_cols}")
    
    return target_columns


# Common NA strings that represent missing values
COMMON_NA_STRINGS = [
    "", " ", "na", "n/a", "n.a.", "n.a", "not available", "not applicable",
    "missing", "null", "nil", "none", "unknown", "undetermined", "undefined",
    "unspecified", "blank", "empty", "-", "--", "---", "?", "??", "???",
    ".", "..", "...", "_", "__", "___", "#n/a", "#na", "#null", "#value!",
    "#error", "#ref!", "#div/0!", "#num!", "#name?", "#null!", "n", "na_character_",
    "na_real_", "na_integer_", "na_complex_", "-99", "-999", "-9999", "99", 
    "999", "9999", "missing data", "no data", "data not available",
    "information not available", "not reported", "not recorded", "not specified",
    "not stated", "not given", "not provided", "not collected", "not applicable",
    "does not apply", "not relevant", "not determined", "not assessed",
    "not evaluated", "not measured", "not tested", "not examined", "not observed"
]


def normalize_text_for_matching(text: Union[str, Any]) -> str:
    """
    Normalize text for case and whitespace insensitive matching.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text string
    """
    if pd.isna(text) or not isinstance(text, str):
        return str(text)
    return str(text).strip().lower()


def validate_dataframe_input(data: Any) -> pd.DataFrame:
    """
    Validate that the input is a pandas DataFrame.
    
    Args:
        data: Input data to validate
        
    Returns:
        Validated DataFrame
        
    Raises:
        TypeError: If input is not a DataFrame
        ValueError: If DataFrame is empty
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(data)}")
    
    if data.empty:
        raise ValueError("Input DataFrame is empty")
    
    if len(data.columns) == 0:
        raise ValueError("Input DataFrame has no columns")
    
    return data


def get_default_params() -> Dict[str, Dict[str, Any]]:
    """
    Get default parameters for data cleaning operations.
    
    Returns:
        Dictionary of default parameters for each cleaning operation
    """
    return {
        "standardize_column_names": {
            "keep": None,
            "rename": None
        },
        "replace_missing_values": {
            "target_columns": None,
            "na_strings": COMMON_NA_STRINGS
        },
        "remove_duplicates": {
            "target_columns": None
        },
        "remove_constants": {
            "cutoff": 1.0
        },
        "standardize_dates": {
            "target_columns": None,
            "error_tolerance": 0.4,
            "format": None,
            "timeframe": None,
            "orders": {
                "world_named_months": ["Ybd", "dby"],
                "world_digit_months": ["dmy", "Ymd"],
                "US_formats": ["Omdy", "YOmd"]
            }
        },
        "standardize_subject_ids": {
            "target_columns": None,
            "prefix": None,
            "suffix": None,
            "range": None,
            "nchar": None
        },
        "to_numeric": {
            "target_columns": None,
            "lang": "en"
        },
        "dictionary": None,
        "check_date_sequence": {
            "target_columns": None
        }
    }


def modify_default_params(default_params: Dict[str, Any], 
                         user_params: Dict[str, Any], 
                         replace_defaults: bool = False) -> Dict[str, Any]:
    """
    Modify default parameters with user-provided parameters.
    
    Args:
        default_params: Default parameters
        user_params: User-provided parameters
        replace_defaults: Whether to replace defaults completely
        
    Returns:
        Modified parameters dictionary
    """
    if replace_defaults:
        return user_params
    
    modified_params = default_params.copy()
    
    for key, value in user_params.items():
        if key in modified_params:
            if isinstance(value, dict) and isinstance(modified_params[key], dict):
                modified_params[key].update(value)
            else:
                modified_params[key] = value
        else:
            modified_params[key] = value
    
    return modified_params