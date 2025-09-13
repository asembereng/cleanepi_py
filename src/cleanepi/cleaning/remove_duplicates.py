"""
Remove duplicate rows functionality.
"""

from typing import List, Optional, Union
import pandas as pd
from loguru import logger

from ..utils.validation import validate_dataframe, validate_columns_exist


def remove_duplicates(
    data: pd.DataFrame,
    target_columns: Optional[List[str]] = None,
    keep: str = "first"
) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
    target_columns : List[str], optional
        Columns to consider for duplicates. If None, use all columns
    keep : str, default "first"
        Which duplicates to keep: 'first', 'last', or False (remove all)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with duplicates removed
    """
    validate_dataframe(data)
    
    if target_columns is not None:
        validate_columns_exist(data, target_columns, "remove_duplicates")
    
    initial_rows = len(data)
    
    # Use pandas drop_duplicates
    result = data.drop_duplicates(subset=target_columns, keep=keep)
    
    duplicates_removed = initial_rows - len(result)
    
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate rows")
    else:
        logger.info("No duplicates found")
    
    return result