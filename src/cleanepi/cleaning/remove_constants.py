"""
Remove constant columns functionality.
"""

from typing import List, Optional
import pandas as pd
from loguru import logger

from ..utils.validation import validate_dataframe


def remove_constants(
    data: pd.DataFrame,
    cutoff: float = 1.0,
    exclude_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Remove columns with constant or near-constant values.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
    cutoff : float, default 1.0
        Proportion of values that must be the same to remove column (0.0-1.0)
    exclude_columns : List[str], optional
        Columns to exclude from constant checking
        
    Returns
    -------
    pd.DataFrame
        DataFrame with constant columns removed
    """
    validate_dataframe(data)
    
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("cutoff must be between 0.0 and 1.0")
    
    exclude_columns = exclude_columns or []
    result = data.copy()
    removed_columns = []
    
    for col in data.columns:
        if col in exclude_columns:
            continue
            
        # Calculate proportion of most common value
        value_counts = data[col].value_counts(dropna=False)
        if len(value_counts) > 0:
            max_count = value_counts.iloc[0]
            proportion = max_count / len(data)
            
            if proportion >= cutoff:
                result = result.drop(columns=[col])
                removed_columns.append(col)
    
    if removed_columns:
        logger.info(f"Removed {len(removed_columns)} constant columns: {removed_columns}")
    else:
        logger.info("No constant columns found")
    
    return result