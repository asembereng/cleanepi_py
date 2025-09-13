"""Replace missing values with NA in pandas DataFrames."""

import pandas as pd
import numpy as np
from typing import List, Optional, Union, Any
from .utils import (
    get_target_column_names, 
    add_to_report, 
    retrieve_column_names,
    normalize_text_for_matching,
    validate_dataframe_input,
    COMMON_NA_STRINGS
)


def replace_missing_values(data: pd.DataFrame,
                          target_columns: Optional[List[str]] = None,
                          na_strings: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Replace missing values with NA in specified columns.
    
    Args:
        data: The input DataFrame
        target_columns: A list of column names. If provided, missing values will be 
                       substituted only in the specified columns. If None, all columns
                       will be processed.
        na_strings: A list of strings that represent the missing values in the columns 
                   of interest. By default, it utilizes common_na_strings. Matching 
                   is insensitive to case and leading/trailing whitespace.
    
    Returns:
        The input DataFrame where missing values are replaced by NA.
        
    Examples:
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        ...     'col1': ['A', '-99', 'C'],
        ...     'col2': ['1', '2', 'missing']
        ... })
        >>> cleaned_data = replace_missing_values(data, na_strings=['-99', 'missing'])
        >>> print(cleaned_data)
    """
    # Validate input
    data = validate_dataframe_input(data)
    
    # Use default NA strings if none provided
    if na_strings is None:
        na_strings = COMMON_NA_STRINGS
    
    # Make a copy to avoid modifying the original data
    data_copy = data.copy()
    
    # Get the correct column names
    target_columns = retrieve_column_names(data_copy, target_columns)
    cols = get_target_column_names(data_copy, target_columns, cols=None)
    
    # Normalize na_strings for case-insensitive matching
    normalized_na_strings = [normalize_text_for_matching(na_str) for na_str in na_strings]
    
    # Track which columns had replacements
    columns_with_replacements = []
    
    # Process each target column
    for col in cols:
        original_values = data_copy[col].copy()
        
        # Only process if column exists
        if col in data_copy.columns:
            # Convert to string for comparison, preserving original non-string types
            temp_col = data_copy[col].astype(str)
            
            # Normalize for comparison
            normalized_col = temp_col.apply(normalize_text_for_matching)
            
            # Find values that match NA strings
            mask = normalized_col.isin(normalized_na_strings)
            
            if mask.any():
                # Replace with NA
                data_copy.loc[mask, col] = np.nan
                columns_with_replacements.append(col)
    
    # Add report information
    if columns_with_replacements:
        data_copy = add_to_report(
            data_copy,
            key="missing_values_replaced_at",
            value=columns_with_replacements
        )
        print(f"Replaced missing values in columns: {columns_with_replacements}")
    else:
        print("Could not detect the provided missing value characters.")
        print("Does your data contain missing value characters other than the specified ones?")
    
    return data_copy


def replace_with_na(series: pd.Series, na_strings: List[str]) -> pd.Series:
    """
    Detect and replace values with NA from a pandas Series.
    
    Args:
        series: A pandas Series of values
        na_strings: A list of values to be replaced with NA
        
    Returns:
        A pandas Series where the specified values were replaced with NA if found.
    """
    # Normalize na_strings for case-insensitive matching
    normalized_na_strings = [normalize_text_for_matching(na_str) for na_str in na_strings]
    
    # Convert series to string for comparison
    temp_series = series.astype(str)
    normalized_series = temp_series.apply(normalize_text_for_matching)
    
    # Create mask for values that should be replaced
    mask = normalized_series.isin(normalized_na_strings)
    
    # Create a copy and replace values
    result = series.copy()
    result.loc[mask] = np.nan
    
    return result