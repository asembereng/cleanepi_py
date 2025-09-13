"""Remove constant data, including empty rows, empty columns, and columns with constant values."""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from .utils import (
    add_to_report,
    validate_dataframe_input
)


def remove_constants(data: pd.DataFrame, cutoff: float = 1.0) -> pd.DataFrame:
    """
    Remove constant data, including empty rows, empty columns, and
    columns with constant values.

    The function iteratively removes constant data until none remain.
    It records details of the removed constant data in the report object.

    Args:
        data: The input DataFrame
        cutoff: A numeric value specifying the cut-off for removing constant data.
               The possible values vary between 0 (excluded) and 1 (included).
               The default is 1 i.e. remove rows and columns with 100% constant data.

    Returns:
        The input dataset where the constant data is filtered out based on
        specified cut-off.
        
    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> data = pd.DataFrame({
        ...     'id': [1, 2, 3, 4],
        ...     'constant_col': [1, 1, 1, 1],
        ...     'empty_col': [np.nan, np.nan, np.nan, np.nan],
        ...     'mixed_col': ['A', 'B', np.nan, 'D']
        ... })
        >>> cleaned_data = remove_constants(data)
        >>> print(cleaned_data)
    """
    # Validate input
    data = validate_dataframe_input(data)
    
    # Validate cutoff
    if not 0 < cutoff <= 1.0:
        if cutoff == 0:
            print("Constant data was not removed. The value for the cut-off argument "
                  "must be greater than 0 and less than or equal to 1.")
            return data
        else:
            raise ValueError("Cutoff must be between 0 (excluded) and 1 (included)")
    
    # Make a copy to avoid modifying the original data
    data_copy = data.copy()
    
    # Perform the constant data removal iteratively
    initial_data = data_copy.copy()
    iteration = 1
    result = _perform_remove_constants(data_copy, cutoff)
    
    # Save details about removed data for report
    constant_data_report = []
    
    # Record first iteration
    constant_data_report.append({
        'iteration': iteration,
        'empty_columns': result['empty_columns'],
        'empty_rows': result['empty_rows'],
        'constant_columns': result['constant_columns']
    })
    
    # Iteratively remove constant data until no more changes
    current_data = result['data']
    while not initial_data.equals(current_data):
        iteration += 1
        temp_data = current_data.copy()
        result = _perform_remove_constants(current_data, cutoff)
        
        # Record this iteration
        constant_data_report.append({
            'iteration': iteration,
            'empty_columns': result['empty_columns'],
            'empty_rows': result['empty_rows'],
            'constant_columns': result['constant_columns']
        })
        
        current_data = result['data']
        
        # Check if no more changes occurred
        if temp_data.equals(current_data):
            break
        
        # Safety check to prevent infinite loops
        if iteration > 100:
            print("Warning: Maximum iterations reached. Stopping constant removal.")
            break
    
    # Add report information
    if len(constant_data_report) > 1:
        print(f"Constant data was removed after {len(constant_data_report)} iterations.")
        print("Enter get_report(data) for more information about what was removed.")
    
    # Add the report to the cleaned data
    current_data = add_to_report(
        current_data,
        key="constant_data",
        value=constant_data_report
    )
    
    return current_data


def _perform_remove_constants(data: pd.DataFrame, cutoff: float) -> Dict[str, Any]:
    """
    Remove constant data in a single iteration.

    This function is called at each iteration of the constant data removal
    process until no constant data remains.

    Args:
        data: The input DataFrame
        cutoff: The cutoff threshold for constant data removal

    Returns:
        A dictionary with the cleaned dataset and information about removed data
    """
    dat = data.copy()
    
    # Track what was removed
    empty_rows = None
    empty_columns = None
    constant_columns = None
    
    # Remove empty rows (rows with >= cutoff proportion of missing values)
    if len(dat) > 0:
        missing_proportion = dat.isnull().sum(axis=1) / len(dat.columns)
        rows_to_remove = missing_proportion >= cutoff
        
        if rows_to_remove.any():
            empty_rows = list(dat.index[rows_to_remove])
            dat = dat[~rows_to_remove]
    
    # Remove empty columns (columns with >= cutoff proportion of missing values)
    if len(dat.columns) > 0 and len(dat) > 0:
        missing_proportion = dat.isnull().sum(axis=0) / len(dat)
        cols_to_remove = missing_proportion >= cutoff
        
        if cols_to_remove.any():
            empty_columns = list(dat.columns[cols_to_remove])
            dat = dat.loc[:, ~cols_to_remove]
    
    # Remove constant columns (columns with only one unique non-null value)
    if len(dat.columns) > 0 and len(dat) > 0:
        constant_cols = []
        for col in dat.columns:
            # Get non-null values
            non_null_values = dat[col].dropna()
            if len(non_null_values) > 0:
                unique_values = non_null_values.nunique()
                if unique_values <= 1:
                    constant_cols.append(col)
        
        if constant_cols:
            constant_columns = constant_cols
            dat = dat.drop(columns=constant_cols)
    
    return {
        'data': dat,
        'empty_columns': empty_columns,
        'empty_rows': empty_rows,
        'constant_columns': constant_columns
    }