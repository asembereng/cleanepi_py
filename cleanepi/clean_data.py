"""Main data cleaning function for cleanepi package."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from .utils import (
    validate_dataframe_input,
    get_default_params,
    modify_default_params
)
from .replace_missing_values import replace_missing_values
from .find_and_remove_duplicates import find_and_remove_duplicates  
from .remove_constants import remove_constants
from .standardize_date import standardize_date
from .standardize_subject_ids import standardize_subject_ids
from .convert_to_numeric import convert_to_numeric
from .print_report import print_report


def clean_data(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Clean and standardize data by performing several operations.
    
    This function cleans up messy DataFrames by performing several operations
    including: cleaning of column names, detecting and removing duplicates,
    empty records and columns, constant columns, replacing missing values by NA,
    converting character columns into dates when they contain a certain number
    of date values, detecting subject IDs with wrong formats, etc.
    
    Args:
        data: The input DataFrame
        **kwargs: Cleaning operations to be applied. Acceptable arguments are:
            - standardize_column_names: Dict with arguments for column name standardization
            - replace_missing_values: Dict with parameters for missing value replacement
            - remove_duplicates: Dict with arguments for duplicate removal
            - remove_constants: Dict with parameters for constant data removal
            - standardize_dates: Dict with parameters for date standardization
            - standardize_subject_ids: Dict with parameters for subject ID checking
            - to_numeric: Dict with parameters for numeric conversion
            - dictionary: DataFrame for dictionary-based cleaning
            - check_date_sequence: Dict for date sequence checking
    
    Returns:
        The cleaned input DataFrame according to the user-specified parameters.
        This is associated with a data cleaning report that can be accessed using
        get_report(cleaned_data) or print_report(cleaned_data)
    
    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from datetime import date
        >>> 
        >>> # Create sample data
        >>> data = pd.DataFrame({
        ...     'study_id': ['PS001P2', 'PS002P2', 'PS003P2', 'PS002P2'],  # duplicate
        ...     'date_column': ['2020-01-01', '28/01/2021', 'Jan 3, 2021', '2021-02-15'],
        ...     'age': ['25', 'thirty', '35', '-99'],  # mixed types, missing value
        ...     'constant_col': [1, 1, 1, 1],  # constant
        ...     'empty_col': [np.nan, np.nan, np.nan, np.nan]  # empty
        ... })
        >>> 
        >>> # Define cleaning parameters
        >>> cleaned_data = clean_data(
        ...     data,
        ...     replace_missing_values={'target_columns': None, 'na_strings': ['-99']},
        ...     remove_duplicates={'target_columns': None},
        ...     remove_constants={'cutoff': 1.0},
        ...     standardize_dates={
        ...         'target_columns': ['date_column'],
        ...         'timeframe': [date(2020, 1, 1), date(2022, 12, 31)]
        ...     },
        ...     to_numeric={'target_columns': ['age'], 'lang': 'en'}
        ... )
        >>> 
        >>> # Print the cleaning report
        >>> print_report(cleaned_data)
    """
    # Validate input
    data = validate_dataframe_input(data)
    
    # Get the default parameters
    default_params = get_default_params()
    
    # Modify the default parameters with the user-provided parameters
    params = modify_default_params(default_params, kwargs, False)
    
    # Make a copy to avoid modifying the original data
    cleaned_data = data.copy()
    
    print("Starting data cleaning process...")
    
    # 1. Standardize column names (simplified - just basic cleanup)
    if params.get("standardize_column_names"):
        print("📝 Cleaning column names")
        cleaned_data = _standardize_column_names(
            cleaned_data,
            keep=params["standardize_column_names"].get("keep"),
            rename=params["standardize_column_names"].get("rename")
        )
    
    # 2. Replace missing values with NA
    if params.get("replace_missing_values"):
        print("🔄 Replacing missing values with NA")
        cleaned_data = replace_missing_values(
            data=cleaned_data,
            target_columns=params["replace_missing_values"].get("target_columns"),
            na_strings=params["replace_missing_values"].get("na_strings")
        )
    
    # 3. Remove constant columns, empty rows and columns
    if params.get("remove_constants"):
        print("🗑️  Removing constant columns and empty rows")
        cleaned_data = remove_constants(
            data=cleaned_data,
            cutoff=params["remove_constants"].get("cutoff", 1.0)
        )
    
    # 4. Remove duplicated rows
    if params.get("remove_duplicates"):
        print("🔍 Removing duplicated rows")
        cleaned_data = find_and_remove_duplicates(
            data=cleaned_data,
            target_columns=params["remove_duplicates"].get("target_columns")
        )
    
    # 5. Standardize Date columns
    if params.get("standardize_dates"):
        print("📅 Standardizing Date columns")
        cleaned_data = standardize_date(
            data=cleaned_data,
            target_columns=params["standardize_dates"].get("target_columns"),
            format=params["standardize_dates"].get("format"),
            timeframe=params["standardize_dates"].get("timeframe"),
            error_tolerance=params["standardize_dates"].get("error_tolerance", 0.4),
            orders=params["standardize_dates"].get("orders")
        )
    
    # 6. Check subject IDs format
    if params.get("standardize_subject_ids"):
        if not params["standardize_subject_ids"].get("target_columns"):
            raise ValueError(
                "You must specify the name of the column that uniquely identifies "
                "the individuals via the 'target_columns' argument."
            )
        
        print("🆔 Checking subject IDs format")
        cleaned_data = standardize_subject_ids(
            data=cleaned_data,
            target_columns=params["standardize_subject_ids"]["target_columns"],
            prefix=params["standardize_subject_ids"].get("prefix"),
            suffix=params["standardize_subject_ids"].get("suffix"),
            range=params["standardize_subject_ids"].get("range"),
            nchar=params["standardize_subject_ids"].get("nchar")
        )
    
    # 7. Convert character values to numeric
    if params.get("to_numeric"):
        target_columns = params["to_numeric"].get("target_columns")
        if target_columns:
            print(f"🔢 Converting the following columns into numeric: {target_columns}")
            cleaned_data = convert_to_numeric(
                data=cleaned_data,
                target_columns=target_columns,
                lang=params["to_numeric"].get("lang", "en")
            )
    
    # 8. Dictionary-based cleaning (placeholder for future implementation)
    if params.get("dictionary") is not None:
        print("📖 Performing dictionary-based cleaning")
        cleaned_data = _clean_using_dictionary(
            cleaned_data, 
            params["dictionary"]
        )
    
    # 9. Check date sequence (placeholder for future implementation)
    if params.get("check_date_sequence"):
        print("📋 Checking whether date sequences are respected")
        cleaned_data = _check_date_sequence(
            cleaned_data,
            target_columns=params["check_date_sequence"].get("target_columns")
        )
    
    print("✅ Data cleaning process completed!")
    
    return cleaned_data


def _standardize_column_names(data: pd.DataFrame, 
                            keep: Optional[List[str]] = None,
                            rename: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Standardize column names using basic cleaning rules.
    
    Args:
        data: Input DataFrame
        keep: List of column names to keep unchanged
        rename: Dictionary mapping old names to new names
        
    Returns:
        DataFrame with standardized column names
    """
    cleaned_data = data.copy()
    
    # Apply custom renaming first
    if rename:
        cleaned_data = cleaned_data.rename(columns=rename)
    
    # Basic column name standardization (snake_case)
    if keep is None:
        keep = []
    
    new_names = {}
    for col in cleaned_data.columns:
        if col not in keep:
            # Convert to snake_case
            new_name = col.lower().replace(' ', '_').replace('.', '_').replace('-', '_')
            # Remove multiple underscores
            while '__' in new_name:
                new_name = new_name.replace('__', '_')
            # Remove leading/trailing underscores
            new_name = new_name.strip('_')
            new_names[col] = new_name
    
    cleaned_data = cleaned_data.rename(columns=new_names)
    
    return cleaned_data


def _clean_using_dictionary(data: pd.DataFrame, dictionary: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for dictionary-based cleaning.
    
    Args:
        data: Input DataFrame
        dictionary: Dictionary DataFrame for value replacement
        
    Returns:
        DataFrame with dictionary-based cleaning applied
    """
    # This is a placeholder implementation
    # In a full implementation, this would use the dictionary to replace coded values
    print("Dictionary-based cleaning is not fully implemented yet.")
    return data


def _check_date_sequence(data: pd.DataFrame, 
                       target_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Placeholder for date sequence checking.
    
    Args:
        data: Input DataFrame  
        target_columns: List of date columns to check sequence for
        
    Returns:
        DataFrame with date sequence check results
    """
    # This is a placeholder implementation
    # In a full implementation, this would check if date sequences are logical
    print("Date sequence checking is not fully implemented yet.")
    return data