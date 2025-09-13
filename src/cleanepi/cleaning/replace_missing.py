"""
Missing value replacement functionality.

Handles various representations of missing values and standardizes them to pandas NA.
"""

from typing import List, Optional, Dict, Any, Union
import pandas as pd
import numpy as np
from loguru import logger

from ..utils.validation import validate_dataframe, validate_columns_exist


def replace_missing_values(
    data: pd.DataFrame,
    target_columns: Optional[List[str]] = None,
    na_strings: Optional[List[str]] = None,
    custom_na_by_column: Optional[Dict[str, List[str]]] = None,
    case_sensitive: bool = False,
    strip_whitespace: bool = True
) -> pd.DataFrame:
    """
    Replace various representations of missing values with pandas NA.
    
    This function identifies and replaces different representations of missing 
    values including common codes like "-99", "N/A", empty strings, etc.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
    target_columns : List[str], optional
        Columns to process. If None, process all columns
    na_strings : List[str], optional
        Global missing value strings to replace
    custom_na_by_column : Dict[str, List[str]], optional
        Column-specific missing value strings
    case_sensitive : bool, default False
        Whether string matching should be case sensitive
    strip_whitespace : bool, default True
        Whether to strip whitespace before comparison
        
    Returns
    -------
    pd.DataFrame
        DataFrame with missing values replaced
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'age': ['25', '-99', '30'],
    ...     'status': ['positive', 'unknown', 'negative']
    ... })
    >>> clean_df = replace_missing_values(df, na_strings=['-99', 'unknown'])
    >>> clean_df.isna().sum()
    age       1
    status    1
    dtype: int64
    """
    validate_dataframe(data)
    
    # Make a copy to avoid modifying original
    result = data.copy()
    
    # Set default NA strings if not provided
    if na_strings is None:
        na_strings = get_default_na_strings()
    
    # Default to all columns if not specified
    if target_columns is None:
        target_columns = list(data.columns)
    else:
        validate_columns_exist(data, target_columns, "replace_missing_values")
    
    custom_na_by_column = custom_na_by_column or {}
    
    logger.info(f"Replacing missing values in {len(target_columns)} columns")
    
    total_replacements = 0
    column_stats = {}
    
    for col in target_columns:
        # Get NA strings for this column
        col_na_strings = na_strings.copy()
        if col in custom_na_by_column:
            col_na_strings.extend(custom_na_by_column[col])
        
        # Remove duplicates while preserving order
        col_na_strings = list(dict.fromkeys(col_na_strings))
        
        # Count replacements for this column
        replacements = _replace_in_column(
            result, col, col_na_strings, case_sensitive, strip_whitespace
        )
        
        total_replacements += replacements
        column_stats[col] = replacements
        
        if replacements > 0:
            logger.debug(f"  {col}: {replacements} values replaced")
    
    logger.info(f"Total missing values replaced: {total_replacements}")
    
    # Log summary statistics
    _log_missing_value_summary(result, column_stats)
    
    return result


def _replace_in_column(
    data: pd.DataFrame,
    column: str,
    na_strings: List[str],
    case_sensitive: bool,
    strip_whitespace: bool
) -> int:
    """
    Replace missing values in a single column.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to modify (in-place)
    column : str
        Column name
    na_strings : List[str]
        Missing value strings
    case_sensitive : bool
        Case sensitivity for matching
    strip_whitespace : bool
        Whether to strip whitespace
        
    Returns
    -------
    int
        Number of values replaced
    """
    if column not in data.columns:
        return 0
    
    original_na_count = data[column].isna().sum()
    
    # Convert column to string for comparison
    col_data = data[column].astype(str)
    
    # Create mask for values to replace
    mask = pd.Series(False, index=data.index)
    
    for na_string in na_strings:
        if strip_whitespace:
            # Compare with whitespace stripped
            if case_sensitive:
                current_mask = col_data.str.strip() == str(na_string).strip()
            else:
                current_mask = col_data.str.strip().str.lower() == str(na_string).strip().lower()
        else:
            # Direct comparison
            if case_sensitive:
                current_mask = col_data == str(na_string)
            else:
                current_mask = col_data.str.lower() == str(na_string).lower()
        
        mask |= current_mask
    
    # Also check for empty strings and whitespace-only strings if strip_whitespace
    if strip_whitespace:
        mask |= col_data.str.strip() == ''
    
    # Replace matched values with NA
    data.loc[mask, column] = pd.NA
    
    # Count new NAs (excluding originally NA values)
    new_na_count = data[column].isna().sum()
    replacements = new_na_count - original_na_count
    
    return max(0, replacements)


def get_default_na_strings() -> List[str]:
    """
    Get default list of strings to treat as missing values.
    
    Returns
    -------
    List[str]
        Default missing value strings
    """
    return [
        # Numeric codes
        "-99", "-999", "99", "999",
        
        # Text representations
        "N/A", "NA", "n/a", "na",
        "NULL", "null", "Null",
        "NIL", "nil", "Nil",
        "MISSING", "missing", "Missing",
        "UNKNOWN", "unknown", "Unknown",
        "UNK", "unk", "Unk",
        
        # Empty and whitespace
        "", " ", "  ", "\t", "\n",
        
        # Common variations
        ".", "..", "...",
        "#N/A", "#NULL", "#REF!",
        "NaN", "nan", "NAN",
        "None", "none", "NONE",
        
        # Language-specific
        "не известно",  # Russian: unknown
        "manquant",     # French: missing
        "desconocido",  # Spanish: unknown
        "sconosciuto",  # Italian: unknown
        "unbekannt",    # German: unknown
        
        # Data entry conventions
        "no data", "no info", "not available",
        "not recorded", "not applicable", "not specified",
        "to be determined", "TBD", "tbd",
        "pending", "PENDING",
    ]


def detect_missing_patterns(data: pd.DataFrame, threshold: float = 0.05) -> Dict[str, Any]:
    """
    Detect potential missing value patterns in the data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
    threshold : float, default 0.05
        Minimum frequency threshold for pattern detection
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with detected patterns and statistics
    """
    validate_dataframe(data)
    
    patterns = {}
    
    for col in data.columns:
        col_patterns = _detect_column_patterns(data[col], threshold)
        if col_patterns:
            patterns[col] = col_patterns
    
    return patterns


def _detect_column_patterns(series: pd.Series, threshold: float) -> Dict[str, Any]:
    """
    Detect missing value patterns in a single column.
    
    Parameters
    ----------
    series : pd.Series
        Column data
    threshold : float
        Frequency threshold
        
    Returns
    -------
    Dict[str, Any]
        Detected patterns for the column
    """
    # Convert to string for analysis
    str_series = series.astype(str)
    
    # Get value counts
    value_counts = str_series.value_counts()
    total_count = len(series)
    
    # Find potential missing patterns
    potential_missing = []
    
    for value, count in value_counts.items():
        frequency = count / total_count
        
        if frequency >= threshold:
            # Check if value looks like missing data
            if _looks_like_missing(value):
                potential_missing.append({
                    'value': value,
                    'count': count,
                    'frequency': frequency
                })
    
    if potential_missing:
        return {
            'potential_missing': potential_missing,
            'total_values': total_count,
            'unique_values': len(value_counts)
        }
    
    return {}


def _looks_like_missing(value: str) -> bool:
    """
    Check if a value looks like it represents missing data.
    
    Parameters
    ----------
    value : str
        Value to check
        
    Returns
    -------
    bool
        True if value looks like missing data
    """
    if pd.isna(value):
        return True
    
    value_str = str(value).strip().lower()
    
    # Check against common missing patterns
    missing_patterns = [
        r'^$',  # Empty string
        r'^\s+$',  # Whitespace only
        r'^-?9+$',  # All 9s (common missing code)
        r'^n/?a$',  # N/A variations
        r'^null$',  # NULL
        r'^missing$',  # MISSING
        r'^unknown$',  # UNKNOWN
        r'^\.+$',  # Dots
        r'^#n/a$',  # Excel NA
    ]
    
    import re
    for pattern in missing_patterns:
        if re.match(pattern, value_str):
            return True
    
    return False


def suggest_na_strings(data: pd.DataFrame, min_frequency: float = 0.01) -> Dict[str, List[str]]:
    """
    Suggest additional NA strings based on data analysis.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
    min_frequency : float, default 0.01
        Minimum frequency to suggest a value as potential NA
        
    Returns
    -------
    Dict[str, List[str]]
        Suggested NA strings by column
    """
    patterns = detect_missing_patterns(data, min_frequency)
    
    suggestions = {}
    for col, col_patterns in patterns.items():
        if 'potential_missing' in col_patterns:
            suggestions[col] = [
                item['value'] for item in col_patterns['potential_missing']
            ]
    
    return suggestions


def _log_missing_value_summary(data: pd.DataFrame, column_stats: Dict[str, int]) -> None:
    """
    Log summary of missing value replacement.
    
    Parameters
    ----------
    data : pd.DataFrame
        Processed DataFrame
    column_stats : Dict[str, int]
        Replacement counts by column
    """
    # Overall missing value statistics
    total_cells = data.size
    total_missing = data.isna().sum().sum()
    missing_percentage = (total_missing / total_cells) * 100
    
    logger.info(f"Missing value summary:")
    logger.info(f"  Total cells: {total_cells:,}")
    logger.info(f"  Missing cells: {total_missing:,} ({missing_percentage:.1f}%)")
    
    # Column-wise statistics
    columns_with_missing = data.isna().sum()
    columns_with_missing = columns_with_missing[columns_with_missing > 0]
    
    if len(columns_with_missing) > 0:
        logger.info(f"  Columns with missing values: {len(columns_with_missing)}")
        for col, missing_count in columns_with_missing.head(10).items():
            percentage = (missing_count / len(data)) * 100
            logger.debug(f"    {col}: {missing_count} ({percentage:.1f}%)")
        
        if len(columns_with_missing) > 10:
            logger.debug(f"    ... and {len(columns_with_missing) - 10} more columns")


def create_missing_value_report(
    original_data: pd.DataFrame,
    cleaned_data: pd.DataFrame,
    na_strings: List[str]
) -> Dict[str, Any]:
    """
    Create a detailed report of missing value replacement.
    
    Parameters
    ----------
    original_data : pd.DataFrame
        Original DataFrame before processing
    cleaned_data : pd.DataFrame
        DataFrame after missing value replacement
    na_strings : List[str]
        NA strings that were used
        
    Returns
    -------
    Dict[str, Any]
        Detailed replacement report
    """
    report = {
        'na_strings_used': na_strings,
        'summary': {
            'total_replacements': 0,
            'columns_affected': 0,
        },
        'by_column': {},
        'by_na_string': {}
    }
    
    total_replacements = 0
    columns_affected = 0
    
    for col in original_data.columns:
        if col in cleaned_data.columns:
            original_missing = original_data[col].isna().sum()
            new_missing = cleaned_data[col].isna().sum()
            replacements = new_missing - original_missing
            
            if replacements > 0:
                total_replacements += replacements
                columns_affected += 1
                
                report['by_column'][col] = {
                    'replacements': replacements,
                    'original_missing': original_missing,
                    'final_missing': new_missing,
                    'missing_percentage': (new_missing / len(cleaned_data)) * 100
                }
    
    report['summary']['total_replacements'] = total_replacements
    report['summary']['columns_affected'] = columns_affected
    
    return report