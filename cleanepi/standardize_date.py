"""Standardize date columns in pandas DataFrames."""

import pandas as pd
import numpy as np
from datetime import datetime, date
from dateutil import parser
from typing import List, Optional, Union, Dict, Any
from .utils import (
    get_target_column_names,
    add_to_report,
    retrieve_column_names,
    validate_dataframe_input
)


def standardize_date(data: pd.DataFrame,
                    target_columns: Optional[List[str]] = None,
                    error_tolerance: float = 0.4,
                    format: Optional[str] = None,
                    timeframe: Optional[List[Union[str, date]]] = None,
                    orders: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
    """
    Standardize date columns in the DataFrame.
    
    Args:
        data: The input DataFrame
        target_columns: A list of column names to standardize. If None, auto-detect date columns.
        error_tolerance: Tolerance for parsing errors (0-1). Default is 0.4.
        format: Specific date format to use. If None, auto-detect.
        timeframe: List of two dates defining the valid timeframe for dates.
        orders: Dictionary defining date parsing orders for different formats.
    
    Returns:
        DataFrame with standardized date columns.
        
    Examples:
        >>> import pandas as pd
        >>> from datetime import date
        >>> data = pd.DataFrame({
        ...     'date_col': ['2020-01-01', '01/02/2020', 'Jan 3, 2020'],
        ...     'other_col': ['A', 'B', 'C']
        ... })
        >>> timeframe = [date(2019, 1, 1), date(2021, 12, 31)]
        >>> cleaned_data = standardize_date(data, timeframe=timeframe)
        >>> print(cleaned_data)
    """
    # Validate input
    data = validate_dataframe_input(data)
    
    # Validate error_tolerance
    if not 0 <= error_tolerance <= 1:
        raise ValueError("error_tolerance must be between 0 and 1")
    
    # Make a copy to avoid modifying the original data
    data_copy = data.copy()
    
    # Get target columns
    if target_columns is None:
        target_columns = detect_date_columns(data_copy)
    
    target_columns = retrieve_column_names(data_copy, target_columns)
    cols = get_target_column_names(data_copy, target_columns, cols=None)
    
    if not cols:
        print("No date columns found or specified.")
        return data_copy
    
    # Set default orders if not provided
    if orders is None:
        orders = {
            "world_named_months": ["Ybd", "dby"],
            "world_digit_months": ["dmy", "Ymd"],
            "US_formats": ["Omdy", "YOmd"]
        }
    
    # Convert timeframe to datetime objects if provided
    if timeframe:
        if len(timeframe) != 2:
            raise ValueError("timeframe must contain exactly 2 dates")
        
        timeframe_dates = []
        for tf_date in timeframe:
            if isinstance(tf_date, str):
                timeframe_dates.append(pd.to_datetime(tf_date).date())
            elif isinstance(tf_date, date):
                timeframe_dates.append(tf_date)
            elif isinstance(tf_date, datetime):
                timeframe_dates.append(tf_date.date())
            else:
                raise ValueError("timeframe dates must be strings, date, or datetime objects")
        timeframe = timeframe_dates
    
    # Process each target column
    standardized_columns = []
    ambiguous_dates = []
    
    for col in cols:
        if col in data_copy.columns:
            original_col = data_copy[col].copy()
            
            # Standardize the date column
            result = standardize_date_column(
                data_copy[col], 
                error_tolerance=error_tolerance,
                format=format,
                timeframe=timeframe,
                orders=orders
            )
            
            data_copy[col] = result['standardized_dates']
            standardized_columns.append(col)
            
            if result['ambiguous_dates']:
                ambiguous_dates.extend([{
                    'column': col,
                    'row': idx,
                    'original_value': val,
                    'possible_formats': formats
                } for idx, val, formats in result['ambiguous_dates']])
    
    # Add report information
    if standardized_columns:
        data_copy = add_to_report(
            data_copy,
            key="standardized_dates",
            value=standardized_columns
        )
    
    if ambiguous_dates:
        data_copy = add_to_report(
            data_copy,
            key="date_standardization_ambiguous",
            value=ambiguous_dates
        )
        print(f"Detected {len(ambiguous_dates)} values that comply with multiple formats.")
        print("Enter get_report(data) to access them.")
    
    print(f"Standardized date columns: {standardized_columns}")
    
    return data_copy


def detect_date_columns(data: pd.DataFrame) -> List[str]:
    """
    Detect columns that likely contain date values.
    
    Args:
        data: Input DataFrame
        
    Returns:
        List of column names that likely contain dates
    """
    date_columns = []
    
    for col in data.columns:
        # Skip if column is already datetime
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            date_columns.append(col)
            continue
        
        # Check if column name suggests it's a date
        col_lower = col.lower()
        date_keywords = ['date', 'time', 'day', 'month', 'year', 'dt_', '_dt', 'dob', 'birth']
        if any(keyword in col_lower for keyword in date_keywords):
            date_columns.append(col)
            continue
        
        # Sample some non-null values to check if they look like dates
        sample_values = data[col].dropna().head(10)
        if len(sample_values) == 0:
            continue
        
        date_like_count = 0
        for value in sample_values:
            if is_date_like(str(value)):
                date_like_count += 1
        
        # If majority of sampled values look like dates, include the column
        if date_like_count / len(sample_values) > 0.6:
            date_columns.append(col)
    
    return date_columns


def is_date_like(value_str: str) -> bool:
    """
    Check if a string value looks like a date.
    
    Args:
        value_str: String value to check
        
    Returns:
        True if the string looks like a date
    """
    if not isinstance(value_str, str) or len(value_str.strip()) == 0:
        return False
    
    # Common date patterns
    date_patterns = [
        r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # YYYY-MM-DD or YYYY/MM/DD
        r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',  # MM-DD-YYYY or MM/DD/YYYY
        r'\d{1,2}[-/]\d{1,2}[-/]\d{2}',  # MM-DD-YY or MM/DD/YY
        r'\w{3}\s+\d{1,2},?\s+\d{4}',    # Mon DD, YYYY
        r'\d{1,2}\s+\w{3}\s+\d{4}',      # DD Mon YYYY
    ]
    
    import re
    for pattern in date_patterns:
        if re.search(pattern, value_str):
            return True
    
    # Try to parse with dateutil as a last resort
    try:
        parser.parse(value_str)
        return True
    except (ValueError, TypeError):
        return False


def standardize_date_column(series: pd.Series,
                          error_tolerance: float = 0.4,
                          format: Optional[str] = None,
                          timeframe: Optional[List[date]] = None,
                          orders: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """
    Standardize a single date column.
    
    Args:
        series: Input pandas Series containing date values
        error_tolerance: Tolerance for parsing errors
        format: Specific date format to use
        timeframe: Valid timeframe for dates
        orders: Date parsing orders
        
    Returns:
        Dictionary containing standardized dates and any ambiguous values
    """
    result_series = series.copy()
    ambiguous_dates = []
    
    for idx, value in series.items():
        if pd.isna(value):
            continue
        
        value_str = str(value).strip()
        if not value_str:
            result_series.iloc[idx] = pd.NaT
            continue
        
        # Try to parse the date
        parsed_date, is_ambiguous, possible_formats = parse_date_value(
            value_str, format=format, timeframe=timeframe, orders=orders
        )
        
        if parsed_date is not None:
            result_series.iloc[idx] = parsed_date
            
            if is_ambiguous:
                ambiguous_dates.append((idx, value_str, possible_formats))
        else:
            result_series.iloc[idx] = pd.NaT
    
    return {
        'standardized_dates': result_series,
        'ambiguous_dates': ambiguous_dates
    }


def parse_date_value(value_str: str,
                    format: Optional[str] = None,
                    timeframe: Optional[List[date]] = None,
                    orders: Optional[Dict[str, List[str]]] = None) -> tuple:
    """
    Parse a date value string into a datetime object.
    
    Args:
        value_str: String value to parse
        format: Specific date format to use
        timeframe: Valid timeframe for dates
        orders: Date parsing orders
        
    Returns:
        Tuple of (parsed_date, is_ambiguous, possible_formats)
    """
    if format:
        # Use specific format
        try:
            parsed_date = pd.to_datetime(value_str, format=format)
            
            # Check timeframe if provided
            if timeframe and not (timeframe[0] <= parsed_date.date() <= timeframe[1]):
                return None, False, []
            
            return parsed_date, False, [format]
        except (ValueError, TypeError):
            return None, False, []
    
    # Try multiple parsing approaches
    possible_formats = []
    parsed_dates = []
    
    # Try pandas to_datetime with infer_datetime_format
    try:
        parsed_date = pd.to_datetime(value_str, infer_datetime_format=True)
        if timeframe is None or (timeframe[0] <= parsed_date.date() <= timeframe[1]):
            parsed_dates.append(parsed_date)
            possible_formats.append('inferred')
    except (ValueError, TypeError):
        pass
    
    # Try dateutil parser
    try:
        parsed_date = pd.to_datetime(parser.parse(value_str))
        if timeframe is None or (timeframe[0] <= parsed_date.date() <= timeframe[1]):
            if parsed_date not in parsed_dates:
                parsed_dates.append(parsed_date)
                possible_formats.append('dateutil')
    except (ValueError, TypeError):
        pass
    
    # Try common date formats
    common_formats = [
        '%Y-%m-%d', '%Y/%m/%d', '%m-%d-%Y', '%m/%d/%Y', '%d-%m-%Y', '%d/%m/%Y',
        '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S', '%d/%m/%Y %H:%M:%S',
        '%b %d, %Y', '%B %d, %Y', '%d %b %Y', '%d %B %Y'
    ]
    
    for fmt in common_formats:
        try:
            parsed_date = pd.to_datetime(value_str, format=fmt)
            if timeframe is None or (timeframe[0] <= parsed_date.date() <= timeframe[1]):
                if parsed_date not in parsed_dates:
                    parsed_dates.append(parsed_date)
                    possible_formats.append(fmt)
        except (ValueError, TypeError):
            continue
    
    if not parsed_dates:
        return None, False, []
    
    # Return the first successful parse, mark as ambiguous if multiple interpretations
    is_ambiguous = len(set(d.date() for d in parsed_dates)) > 1
    
    return parsed_dates[0], is_ambiguous, possible_formats


# Alias for consistency
standardize_dates = standardize_date