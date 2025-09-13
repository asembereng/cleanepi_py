"""Standardize subject IDs in pandas DataFrames."""

import pandas as pd
import numpy as np
import re
from typing import List, Optional, Union, Tuple
from .utils import (
    get_target_column_names,
    add_to_report,
    retrieve_column_names,
    validate_dataframe_input
)


def standardize_subject_ids(data: pd.DataFrame,
                          target_columns: List[str],
                          prefix: Optional[str] = None,
                          suffix: Optional[str] = None,
                          range: Optional[Tuple[int, int]] = None,
                          nchar: Optional[int] = None) -> pd.DataFrame:
    """
    Check and standardize subject IDs format.
    
    Args:
        data: The input DataFrame
        target_columns: List of column names containing subject IDs
        prefix: Expected prefix for subject IDs
        suffix: Expected suffix for subject IDs  
        range: Tuple of (min, max) for numeric part of IDs
        nchar: Expected total character length of IDs
        
    Returns:
        DataFrame with checked subject IDs and report information
        
    Examples:
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        ...     'study_id': ['PS001P2', 'PS002P2', 'XS003P2', 'PS004P2'],
        ...     'other_col': ['A', 'B', 'C', 'D']
        ... })
        >>> cleaned_data = standardize_subject_ids(
        ...     data, target_columns=['study_id'], 
        ...     prefix='PS', suffix='P2', range=(1, 100), nchar=7
        ... )
        >>> print(cleaned_data)
    """
    # Validate input
    data = validate_dataframe_input(data)
    
    if not target_columns:
        raise ValueError("target_columns must be specified for subject ID checking")
    
    # Make a copy to avoid modifying the original data
    data_copy = data.copy()
    
    # Get the correct column names
    target_columns = retrieve_column_names(data_copy, target_columns)
    cols = get_target_column_names(data_copy, target_columns, cols=None)
    
    # Check each target column
    all_missing = []
    all_duplicates = []
    all_incorrect = []
    
    for col in cols:
        if col in data_copy.columns:
            missing, duplicates, incorrect = check_subject_ids_column(
                data_copy[col], prefix=prefix, suffix=suffix, 
                range=range, nchar=nchar
            )
            
            all_missing.extend([(col, idx) for idx in missing])
            all_duplicates.extend([(col, val, indices) for val, indices in duplicates])
            all_incorrect.extend([(col, idx, val) for idx, val in incorrect])
    
    # Add report information
    if all_missing or all_duplicates or all_incorrect:
        print(f"Detected {len(all_missing)} missing, {len(all_duplicates)} duplicated, "
              f"and {len(all_incorrect)} incorrect subject IDs.")
        
        if all_incorrect:
            print("Enter get_report(data) to access them.")
            print("You can use the correct_subject_ids() function to correct them.")
    
    data_copy = add_to_report(
        data_copy,
        key="subject_id_check",
        value={
            'missing_ids': all_missing,
            'duplicate_ids': all_duplicates,
            'incorrect_ids': all_incorrect
        }
    )
    
    return data_copy


def check_subject_ids_column(series: pd.Series,
                           prefix: Optional[str] = None,
                           suffix: Optional[str] = None,
                           range: Optional[Tuple[int, int]] = None,
                           nchar: Optional[int] = None) -> Tuple[List, List, List]:
    """
    Check subject IDs in a single column.
    
    Args:
        series: Pandas Series containing subject IDs
        prefix: Expected prefix
        suffix: Expected suffix
        range: Expected range for numeric part
        nchar: Expected character length
        
    Returns:
        Tuple of (missing_indices, duplicates, incorrect_ids)
    """
    missing_indices = []
    duplicates = []
    incorrect_ids = []
    
    # Find missing values
    missing_mask = series.isnull() | (series == '') | (series.astype(str).str.strip() == '')
    missing_indices = series.index[missing_mask].tolist()
    
    # Check for duplicates among non-missing values
    non_missing = series[~missing_mask]
    if len(non_missing) > 0:
        value_counts = non_missing.value_counts()
        duplicate_values = value_counts[value_counts > 1]
        
        for value, count in duplicate_values.items():
            indices = series[series == value].index.tolist()
            duplicates.append((value, indices))
    
    # Check format compliance
    for idx, value in series.items():
        if pd.isna(value) or str(value).strip() == '':
            continue
            
        value_str = str(value).strip()
        
        if not is_id_format_correct(value_str, prefix, suffix, range, nchar):
            incorrect_ids.append((idx, value_str))
    
    return missing_indices, duplicates, incorrect_ids


def is_id_format_correct(id_str: str,
                        prefix: Optional[str] = None,
                        suffix: Optional[str] = None,
                        range: Optional[Tuple[int, int]] = None,
                        nchar: Optional[int] = None) -> bool:
    """
    Check if a subject ID string matches the expected format.
    
    Args:
        id_str: Subject ID string to check
        prefix: Expected prefix
        suffix: Expected suffix
        range: Expected range for numeric part
        nchar: Expected character length
        
    Returns:
        True if ID format is correct
    """
    # Check character length
    if nchar and len(id_str) != nchar:
        return False
    
    # Check prefix
    if prefix and not id_str.startswith(prefix):
        return False
    
    # Check suffix
    if suffix and not id_str.endswith(suffix):
        return False
    
    # Extract numeric part if prefix/suffix specified
    if prefix or suffix:
        start_idx = len(prefix) if prefix else 0
        end_idx = len(id_str) - len(suffix) if suffix else len(id_str)
        
        if start_idx >= end_idx:
            return False
            
        numeric_part = id_str[start_idx:end_idx]
        
        # Check if numeric part is actually numeric
        try:
            numeric_value = int(numeric_part)
        except ValueError:
            return False
        
        # Check range
        if range and not (range[0] <= numeric_value <= range[1]):
            return False
    
    return True


def correct_subject_ids(data: pd.DataFrame,
                       target_columns: List[str],
                       corrections: dict) -> pd.DataFrame:
    """
    Correct subject IDs based on provided corrections.
    
    Args:
        data: Input DataFrame
        target_columns: List of columns containing subject IDs
        corrections: Dictionary mapping incorrect IDs to correct ones
        
    Returns:
        DataFrame with corrected subject IDs
    """
    data_copy = data.copy()
    
    for col in target_columns:
        if col in data_copy.columns:
            data_copy[col] = data_copy[col].replace(corrections)
    
    return data_copy


# Alias for consistency  
check_subject_ids = standardize_subject_ids