"""Date standardization functionality."""

from typing import List, Optional, Dict, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, date
import re
from loguru import logger

from ..utils.validation import validate_dataframe, validate_columns_exist


def detect_date_columns(data: pd.DataFrame) -> List[str]:
    """
    Auto-detect columns that likely contain dates.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
        
    Returns
    -------
    List[str]
        List of column names that likely contain dates
    """
    date_columns = []
    
    for col in data.columns:
        col_lower = str(col).lower()
        # Check for common date column name patterns
        date_patterns = [
            'date', 'time', 'birth', 'created', 'updated', 'start', 'end',
            'admission', 'discharge', 'visit', 'onset', 'diagnosis'
        ]
        
        if any(pattern in col_lower for pattern in date_patterns):
            date_columns.append(col)
            continue
            
        # Check if column contains date-like strings
        sample_size = min(100, len(data))
        sample_data = data[col].dropna().head(sample_size)
        
        if len(sample_data) == 0:
            continue
            
        # Convert to string and check for date patterns
        sample_strings = sample_data.astype(str)
        date_like_count = 0
        
        for value in sample_strings:
            if _is_date_like(value):
                date_like_count += 1
                
        # If more than 50% of values look like dates, consider it a date column
        if date_like_count / len(sample_strings) > 0.5:
            date_columns.append(col)
    
    return date_columns


def _is_date_like(value: str) -> bool:
    """Check if a string value looks like a date."""
    if not isinstance(value, str) or len(value.strip()) == 0:
        return False
        
    # Common date patterns
    date_patterns = [
        r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # YYYY-MM-DD or YYYY/MM/DD
        r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',  # MM-DD-YYYY or MM/DD/YYYY or DD-MM-YYYY
        r'\d{1,2}[-/]\d{1,2}[-/]\d{2}',  # MM-DD-YY or DD-MM-YY
        r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',  # DD Mon
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}',  # Mon DD
    ]
    
    for pattern in date_patterns:
        if re.search(pattern, value, re.IGNORECASE):
            return True
            
    return False


def get_default_date_formats() -> List[str]:
    """Get default date formats for parsing."""
    return [
        '%Y-%m-%d',          # 2023-01-15
        '%Y/%m/%d',          # 2023/01/15
        '%d-%m-%Y',          # 15-01-2023
        '%d/%m/%Y',          # 15/01/2023
        '%m-%d-%Y',          # 01-15-2023
        '%m/%d/%Y',          # 01/15/2023
        '%Y-%m-%d %H:%M:%S', # 2023-01-15 14:30:00
        '%Y/%m/%d %H:%M:%S', # 2023/01/15 14:30:00
        '%d-%m-%Y %H:%M:%S', # 15-01-2023 14:30:00
        '%d/%m/%Y %H:%M:%S', # 15/01/2023 14:30:00
        '%m-%d-%Y %H:%M:%S', # 01-15-2023 14:30:00
        '%m/%d/%Y %H:%M:%S', # 01/15/2023 14:30:00
        '%Y-%m-%d %H:%M',    # 2023-01-15 14:30
        '%Y/%m/%d %H:%M',    # 2023/01/15 14:30
        '%d-%m-%Y %H:%M',    # 15-01-2023 14:30
        '%d/%m/%Y %H:%M',    # 15/01/2023 14:30
        '%m-%d-%Y %H:%M',    # 01-15-2023 14:30
        '%m/%d/%Y %H:%M',    # 01/15/2023 14:30
        '%d %b %Y',          # 15 Jan 2023
        '%b %d, %Y',         # Jan 15, 2023
        '%B %d, %Y',         # January 15, 2023
        '%d %B %Y',          # 15 January 2023
        '%Y%m%d',            # 20230115
        '%d%m%Y',            # 15012023
    ]


def standardize_dates(
    data: pd.DataFrame,
    target_columns: Optional[List[str]] = None,
    formats: Optional[List[str]] = None,
    timeframe: Optional[Tuple[str, str]] = None,
    error_tolerance: float = 0.4,
    orders: Optional[Dict[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Standardize date columns with intelligent parsing.
    
    This function detects and standardizes date columns by:
    - Auto-detecting date columns if not specified
    - Trying multiple date formats for parsing
    - Validating dates against timeframes
    - Handling various date representations
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
    target_columns : List[str], optional
        Columns to standardize. If None, auto-detect date columns
    formats : List[str], optional
        Date formats to try. If None, use comprehensive default list
    timeframe : Tuple[str, str], optional
        Valid date range as (start_date, end_date) in YYYY-MM-DD format
    error_tolerance : float, default 0.4
        Proportion of unparseable dates to tolerate (0.0-1.0)
    orders : Dict[str, List[str]], optional
        Custom parsing orders by category (future enhancement)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with standardized date columns
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'date_of_birth': ['1990-01-15', '15/01/1990', '01-15-1990'],
    ...     'visit_date': ['2023/03/20', '20-03-2023', '03/20/2023']
    ... })
    >>> clean_df = standardize_dates(df)
    >>> clean_df.dtypes
    date_of_birth    datetime64[ns]
    visit_date       datetime64[ns]
    dtype: object
    """
    validate_dataframe(data)
    
    # Make a copy to avoid modifying original
    result = data.copy()
    
    # Auto-detect date columns if not specified
    if target_columns is None:
        target_columns = detect_date_columns(data)
        if target_columns:
            logger.info(f"Auto-detected date columns: {target_columns}")
        else:
            logger.info("No date columns detected")
            return result
    else:
        validate_columns_exist(data, target_columns, "standardize_dates")
    
    if not target_columns:
        logger.info("No date columns to process")
        return result
    
    # Use default formats if not provided
    if formats is None:
        formats = get_default_date_formats()
    
    # Parse timeframe if provided
    timeframe_start = None
    timeframe_end = None
    if timeframe is not None:
        try:
            timeframe_start = pd.to_datetime(timeframe[0])
            timeframe_end = pd.to_datetime(timeframe[1])
            logger.info(f"Using timeframe: {timeframe[0]} to {timeframe[1]}")
        except Exception as e:
            logger.warning(f"Invalid timeframe format: {e}")
    
    logger.info(f"Standardizing {len(target_columns)} date columns")
    
    for col in target_columns:
        logger.info(f"Processing date column: {col}")
        
        # Skip if already datetime
        if pd.api.types.is_datetime64_any_dtype(result[col]):
            logger.info(f"Column {col} is already datetime, skipping")
            continue
        
        original_values = result[col].copy()
        total_values = len(original_values.dropna())
        
        if total_values == 0:
            logger.warning(f"Column {col} has no non-null values")
            continue
        
        # Try to parse dates
        parsed_dates = _parse_date_column(original_values, formats)
        
        # Check error tolerance
        successful_parses = parsed_dates.notna().sum()
        error_rate = 1.0 - (successful_parses / total_values)
        
        if error_rate > error_tolerance:
            logger.warning(
                f"Column {col}: {error_rate:.1%} parsing errors exceed tolerance "
                f"of {error_tolerance:.1%}, skipping standardization"
            )
            continue
        
        # Apply timeframe validation if specified
        if timeframe_start is not None and timeframe_end is not None:
            valid_dates = parsed_dates[
                (parsed_dates >= timeframe_start) & 
                (parsed_dates <= timeframe_end)
            ]
            invalid_count = len(parsed_dates.dropna()) - len(valid_dates)
            
            if invalid_count > 0:
                logger.warning(
                    f"Column {col}: {invalid_count} dates outside timeframe"
                )
                # Set invalid dates to NaT
                parsed_dates = parsed_dates.where(
                    (parsed_dates >= timeframe_start) & 
                    (parsed_dates <= timeframe_end)
                )
        
        # Update the result
        result[col] = parsed_dates
        
        success_count = parsed_dates.notna().sum()
        logger.info(
            f"Column {col}: Successfully parsed {success_count}/{total_values} "
            f"dates ({success_count/total_values:.1%})"
        )
    
    return result


def _parse_date_column(series: pd.Series, formats: List[str]) -> pd.Series:
    """
    Parse a series using multiple date formats.
    
    Parameters
    ----------
    series : pd.Series
        Series containing date strings
    formats : List[str]
        Date formats to try
        
    Returns
    -------
    pd.Series
        Series with parsed dates
    """
    result = pd.Series(index=series.index, dtype='datetime64[ns]')
    remaining_mask = series.notna() & (result.isna())
    
    # First try pandas built-in date parsing (most flexible)
    try:
        parsed = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
        result = result.fillna(parsed)
        remaining_mask = series.notna() & (result.isna())
        
        if not remaining_mask.any():
            return result
    except Exception:
        pass
    
    # Try each format explicitly
    for fmt in formats:
        if not remaining_mask.any():
            break
            
        try:
            remaining_series = series[remaining_mask]
            parsed = pd.to_datetime(remaining_series, format=fmt, errors='coerce')
            
            # Update result with successfully parsed dates
            valid_parsed = parsed.notna()
            if valid_parsed.any():
                result.loc[remaining_mask] = result.loc[remaining_mask].fillna(parsed)
                remaining_mask = series.notna() & (result.isna())
                
        except Exception:
            continue
    
    return result