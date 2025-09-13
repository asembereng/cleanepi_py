"""Guess date formats in pandas DataFrames."""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from .utils import validate_dataframe_input
from .standardize_date import parse_date_value


def guess_dates(data: pd.DataFrame,
                target_columns: Optional[List[str]] = None,
                threshold: float = 0.7) -> pd.DataFrame:
    """
    Guess date formats in specified columns.
    
    Args:
        data: Input DataFrame
        target_columns: List of columns to check for dates
        threshold: Minimum proportion of values that must be date-like
        
    Returns:
        DataFrame with guessed date information in report
        
    Examples:
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        ...     'mixed_col': ['2020-01-01', 'not a date', '01/02/2020', 'text'],
        ...     'date_col': ['Jan 1, 2020', 'Feb 2, 2020', 'Mar 3, 2020']
        ... })
        >>> result = guess_dates(data)
        >>> print(result)
    """
    # Validate input
    data = validate_dataframe_input(data)
    
    # Make a copy to avoid modifying the original data
    data_copy = data.copy()
    
    # If no target columns specified, check all columns
    if target_columns is None:
        target_columns = list(data_copy.columns)
    
    # Guess dates for each column
    date_guesses = {}
    
    for col in target_columns:
        if col in data_copy.columns:
            guess_result = guess_date_column(data_copy[col], threshold=threshold)
            if guess_result['is_likely_date']:
                date_guesses[col] = guess_result
    
    # Add to report
    if hasattr(data_copy, '_cleanepi_report'):
        data_copy._cleanepi_report['date_guesses'] = date_guesses
    else:
        data_copy._cleanepi_report = {'date_guesses': date_guesses}
    
    if date_guesses:
        print(f"Found potential date columns: {list(date_guesses.keys())}")
    
    return data_copy


def guess_date_column(series: pd.Series, threshold: float = 0.7) -> Dict[str, Any]:
    """
    Guess if a column contains dates and what format they might be in.
    
    Args:
        series: Input pandas Series
        threshold: Minimum proportion of values that must be date-like
        
    Returns:
        Dictionary with date guessing results
    """
    non_null_values = series.dropna()
    if len(non_null_values) == 0:
        return {'is_likely_date': False, 'confidence': 0.0, 'formats': []}
    
    # Count how many values can be parsed as dates
    successful_parses = 0
    identified_formats = []
    
    for value in non_null_values.head(min(100, len(non_null_values))):  # Sample for performance
        value_str = str(value).strip()
        if not value_str:
            continue
            
        parsed_date, is_ambiguous, possible_formats = parse_date_value(value_str)
        if parsed_date is not None:
            successful_parses += 1
            identified_formats.extend(possible_formats)
    
    # Calculate confidence
    sample_size = min(100, len(non_null_values))
    confidence = successful_parses / sample_size if sample_size > 0 else 0.0
    
    # Determine most common formats
    from collections import Counter
    format_counts = Counter(identified_formats)
    common_formats = [fmt for fmt, count in format_counts.most_common(3)]
    
    return {
        'is_likely_date': confidence >= threshold,
        'confidence': confidence,
        'formats': common_formats,
        'sample_size': sample_size,
        'successful_parses': successful_parses
    }