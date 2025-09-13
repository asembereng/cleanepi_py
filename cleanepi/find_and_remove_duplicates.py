"""Find and remove duplicate rows from pandas DataFrames."""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from .utils import (
    get_target_column_names,
    add_to_report,
    retrieve_column_names,
    validate_dataframe_input
)


def find_and_remove_duplicates(data: pd.DataFrame,
                              target_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicates from the DataFrame.
    
    When removing duplicates, users can specify a set of columns to consider with
    the target_columns argument.
    
    Args:
        data: The input DataFrame
        target_columns: A list of column names to use when looking for duplicates.
                       Default is None (considers all columns).
    
    Returns:
        The input DataFrame without the duplicated rows identified from all or 
        the specified columns.
        
    Note:
        Caveat: In many epidemiological datasets, multiple rows may share the
        same value in one or more columns without being true duplicates.
        For example, several individuals might have the same symptom onset date
        and admission date. Be cautious when using this function—especially when
        applying it to a single target column—to avoid incorrect identification
        or removal of valid entries.
        
    Examples:
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        ...     'id': [1, 2, 2, 3],
        ...     'name': ['A', 'B', 'B', 'C'],
        ...     'value': [10, 20, 20, 30]
        ... })
        >>> cleaned_data = find_and_remove_duplicates(data)
        >>> print(cleaned_data)
    """
    # Validate input
    data = validate_dataframe_input(data)
    
    # Make a copy to avoid modifying the original data
    data_copy = data.copy()
    
    # Get the correct column names
    target_columns = retrieve_column_names(data_copy, target_columns)
    cols = get_target_column_names(data_copy, target_columns, cols=None)
    
    # Find duplicates first
    duplicates_info = find_duplicates(data_copy, target_columns)
    duplicate_report = duplicates_info.get('found_duplicates', {})
    
    # Check if duplicates were found
    if 'duplicated_rows' in duplicate_report and len(duplicate_report['duplicated_rows']) > 0:
        # Add row_id for tracking
        data_copy = data_copy.reset_index(drop=True)
        data_copy['row_id'] = range(len(data_copy))
        
        # Remove duplicates keeping the first occurrence
        if cols:
            data_cleaned = data_copy.drop_duplicates(subset=cols, keep='first')
        else:
            data_cleaned = data_copy.drop_duplicates(keep='first')
        
        # Identify removed rows
        removed_rows = data_copy[~data_copy.index.isin(data_cleaned.index)]
        
        # Remove the temporary row_id column
        data_cleaned = data_cleaned.drop('row_id', axis=1)
        
        # Add report about removed duplicates
        removed_info = removed_rows[['row_id'] + cols].to_dict('records')
        data_cleaned = add_to_report(
            data_cleaned,
            key="removed_duplicates",
            value=removed_info
        )
        
        # Also add the original duplicate findings
        data_cleaned = add_to_report(
            data_cleaned,
            key="found_duplicates",
            value=duplicate_report
        )
        
        print(f"Removed {len(removed_rows)} duplicate rows.")
        
    else:
        print("No duplicates were found.")
        data_cleaned = data_copy
    
    return data_cleaned


def find_duplicates(data: pd.DataFrame,
                   target_columns: Optional[List[str]] = None) -> dict:
    """
    Identify and return information about duplicated rows in a DataFrame.
    
    Args:
        data: The input DataFrame
        target_columns: A list of column names to consider when looking for duplicates.
                       Default is None (considers all columns).
    
    Returns:
        A dictionary containing information about found duplicates with keys:
        - 'duplicated_rows': List of dictionaries containing duplicate row information
        - 'duplicates_checked_from': List of columns that were checked for duplicates
        
    Examples:
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        ...     'id': [1, 2, 2, 3],
        ...     'name': ['A', 'B', 'B', 'C'],
        ...     'value': [10, 20, 20, 30]
        ... })
        >>> dups_info = find_duplicates(data)
        >>> print(dups_info)
    """
    # Validate input
    data = validate_dataframe_input(data)
    
    # Get the correct column names
    target_columns = retrieve_column_names(data, target_columns)
    cols = get_target_column_names(data, target_columns, cols=None)
    
    # Add row_id for tracking original positions
    data_with_rowid = data.copy()
    data_with_rowid['row_id'] = range(len(data_with_rowid))
    
    # Find duplicates
    if cols:
        # Check for duplicates in specified columns
        duplicate_mask = data_with_rowid.duplicated(subset=cols, keep=False)
    else:
        # Check for duplicates across all columns
        duplicate_mask = data_with_rowid.duplicated(keep=False)
    
    duplicates_df = data_with_rowid[duplicate_mask].copy()
    
    if len(duplicates_df) > 0:
        # Sort by the target columns for better organization
        if cols:
            duplicates_df = duplicates_df.sort_values(cols)
        
        # Add group_id to identify duplicate groups
        if cols:
            duplicates_df['group_id'] = duplicates_df.groupby(cols).ngroup() + 1
        else:
            # If checking all columns, each set of identical rows gets same group_id
            duplicates_df['group_id'] = duplicates_df.groupby(list(data.columns)).ngroup() + 1
        
        # Reorder columns to have row_id and group_id first
        id_cols = ['row_id', 'group_id']
        other_cols = [col for col in duplicates_df.columns if col not in id_cols]
        duplicates_df = duplicates_df[id_cols + other_cols]
        
        print(f"Found {len(duplicates_df)} duplicated rows in the dataset.")
        
        # Return report information
        duplicates_report = {
            'duplicated_rows': duplicates_df.to_dict('records'),
            'duplicates_checked_from': cols
        }
        
        return {'found_duplicates': duplicates_report}
    
    else:
        print("No duplicates were found.")
        return {'found_duplicates': {'duplicated_rows': [], 'duplicates_checked_from': cols}}


# Alias for consistency with main function naming
remove_duplicates = find_and_remove_duplicates