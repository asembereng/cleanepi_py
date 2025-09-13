"""Dictionary-based cleaning functionality."""

from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from loguru import logger

from ..utils.validation import validate_dataframe, validate_columns_exist


def clean_using_dictionary(
    data: pd.DataFrame,
    dictionary: Dict[str, Dict[str, str]],
    case_sensitive: bool = False,
    exact_match: bool = True,
    default_action: str = "keep"
) -> pd.DataFrame:
    """
    Clean data using dictionary mappings for value replacement.
    
    This function applies dictionary-based cleaning by:
    - Replacing coded values with meaningful labels
    - Standardizing categorical values
    - Handling case sensitivity options
    - Supporting partial matching
    - Providing detailed mapping reports
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
    dictionary : Dict[str, Dict[str, str]]
        Dictionary mapping for each column: {column_name: {old_value: new_value}}
    case_sensitive : bool, default False
        Whether string matching should be case sensitive
    exact_match : bool, default True
        Whether to require exact matches or allow partial matching
    default_action : str, default "keep"
        Action for values not found in dictionary:
        - "keep": keep original values
        - "null": convert to NaN
        - "flag": add suffix "_unmapped"
        
    Returns
    -------
    pd.DataFrame
        DataFrame with values replaced according to dictionary
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'status': ['pos', 'neg', 'positive', 'negative'],
    ...     'gender': ['M', 'F', 'm', 'f'],
    ...     'result': [1, 0, 1, 0]
    ... })
    >>> dictionary = {
    ...     'status': {'pos': 'positive', 'neg': 'negative'},
    ...     'gender': {'M': 'Male', 'F': 'Female', 'm': 'Male', 'f': 'Female'},
    ...     'result': {'1': 'positive', '0': 'negative'}
    ... }
    >>> cleaned = clean_using_dictionary(df, dictionary)
    >>> cleaned['status'].tolist()
    ['positive', 'negative', 'positive', 'negative']
    """
    validate_dataframe(data)
    
    # Validate dictionary structure
    if not isinstance(dictionary, dict):
        raise ValueError("dictionary must be a dict")
    
    for col, mappings in dictionary.items():
        if not isinstance(mappings, dict):
            raise ValueError(f"Mappings for column '{col}' must be a dict")
    
    # Check that specified columns exist
    specified_columns = list(dictionary.keys())
    validate_columns_exist(data, specified_columns, "clean_using_dictionary")
    
    # Make a copy to avoid modifying original
    result = data.copy()
    
    logger.info(f"Applying dictionary cleaning to {len(specified_columns)} columns")
    
    cleaning_summary = {}
    
    for col, mappings in dictionary.items():
        logger.info(f"Processing column: {col}")
        
        if col not in result.columns:
            logger.warning(f"Column '{col}' not found in data, skipping")
            continue
        
        original_series = result[col].copy()
        total_values = len(original_series.dropna())
        
        if total_values == 0:
            logger.warning(f"Column {col} has no non-null values")
            cleaning_summary[col] = {
                'total_values': 0,
                'mapped_values': 0,
                'unmapped_values': 0,
                'mapping_rate': 0.0
            }
            continue
        
        # Apply mappings
        mapped_series, mapping_stats = _apply_column_mapping(
            original_series, mappings, case_sensitive, exact_match, default_action
        )
        
        # Update result
        result[col] = mapped_series
        
        # Store summary
        cleaning_summary[col] = mapping_stats
        
        logger.info(
            f"Column {col}: {mapping_stats['mapped_values']}/{total_values} values mapped "
            f"({mapping_stats['mapping_rate']:.1%} mapping rate)"
        )
        
        # Log unmapped values if any
        if mapping_stats['unmapped_values'] > 0:
            unmapped_sample = mapping_stats.get('unmapped_sample', [])
            if unmapped_sample:
                logger.info(f"Sample unmapped values in {col}: {unmapped_sample[:5]}")
    
    # Log overall summary
    total_mappings = sum(s['mapped_values'] for s in cleaning_summary.values())
    total_values_all = sum(s['total_values'] for s in cleaning_summary.values())
    
    if total_values_all > 0:
        logger.info(
            f"Overall: {total_mappings}/{total_values_all} values mapped "
            f"({(total_mappings/total_values_all):.1%} success rate)"
        )
    
    return result


def _apply_column_mapping(
    series: pd.Series,
    mappings: Dict[str, str],
    case_sensitive: bool,
    exact_match: bool,
    default_action: str
) -> tuple[pd.Series, Dict[str, Any]]:
    """
    Apply mapping to a single column.
    
    Parameters
    ----------
    series : pd.Series
        Series to map
    mappings : Dict[str, str]
        Value mappings
    case_sensitive : bool
        Whether to use case sensitive matching
    exact_match : bool
        Whether to require exact matches
    default_action : str
        Action for unmapped values
        
    Returns
    -------
    tuple[pd.Series, Dict[str, Any]]
        Mapped series and mapping statistics
    """
    result_series = series.copy()
    mapped_count = 0
    unmapped_values = set()
    
    # Prepare mappings for case-insensitive matching if needed
    if not case_sensitive:
        mappings_lower = {k.lower(): v for k, v in mappings.items()}
    else:
        mappings_lower = mappings
    
    for idx, value in series.items():
        if pd.isna(value):
            continue
        
        str_value = str(value)
        mapped_value = None
        
        # Try exact match first
        if exact_match:
            if case_sensitive:
                mapped_value = mappings.get(str_value)
            else:
                mapped_value = mappings_lower.get(str_value.lower())
        else:
            # Try partial matching
            mapped_value = _find_partial_match(str_value, mappings, case_sensitive)
        
        if mapped_value is not None:
            result_series.iloc[idx] = mapped_value
            mapped_count += 1
        else:
            # Handle unmapped values
            unmapped_values.add(str_value)
            
            if default_action == "null":
                result_series.iloc[idx] = np.nan
            elif default_action == "flag":
                result_series.iloc[idx] = f"{str_value}_unmapped"
            # For "keep", do nothing (keep original value)
    
    # Calculate statistics
    total_non_null = len(series.dropna())
    unmapped_count = total_non_null - mapped_count
    mapping_rate = mapped_count / total_non_null if total_non_null > 0 else 0.0
    
    stats = {
        'total_values': total_non_null,
        'mapped_values': mapped_count,
        'unmapped_values': unmapped_count,
        'mapping_rate': mapping_rate,
        'unmapped_sample': list(unmapped_values)[:10]  # Sample of unmapped values
    }
    
    return result_series, stats


def _find_partial_match(
    value: str,
    mappings: Dict[str, str],
    case_sensitive: bool
) -> Optional[str]:
    """
    Find partial matches in mappings.
    
    Parameters
    ----------
    value : str
        Value to match
    mappings : Dict[str, str]
        Available mappings
    case_sensitive : bool
        Whether to use case sensitive matching
        
    Returns
    -------
    str or None
        Mapped value if found, None otherwise
    """
    if not case_sensitive:
        value = value.lower()
    
    # Try substring matching
    for key, mapped_val in mappings.items():
        key_compare = key if case_sensitive else key.lower()
        
        # Check if key is substring of value or vice versa
        if key_compare in value or value in key_compare:
            return mapped_val
    
    return None


def create_mapping_dictionary(
    data: pd.DataFrame,
    columns: List[str],
    include_counts: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Create a template mapping dictionary from existing data.
    
    This function analyzes categorical columns and creates a template
    dictionary that can be modified for cleaning purposes.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
    columns : List[str]
        Columns to analyze
    include_counts : bool, default True
        Whether to include value counts in the template
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Template dictionary with unique values and optional counts
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'status': ['pos', 'neg', 'pos', 'negative'],
    ...     'gender': ['M', 'F', 'm', 'F']
    ... })
    >>> template = create_mapping_dictionary(df, ['status', 'gender'])
    >>> template['status']
    {'pos': {'suggested_mapping': 'pos', 'count': 2},
     'neg': {'suggested_mapping': 'neg', 'count': 1},
     'negative': {'suggested_mapping': 'negative', 'count': 1}}
    """
    validate_dataframe(data)
    validate_columns_exist(data, columns, "create_mapping_dictionary")
    
    template = {}
    
    for col in columns:
        logger.info(f"Analyzing column: {col}")
        
        # Get value counts
        value_counts = data[col].value_counts(dropna=True)
        
        col_template = {}
        for value, count in value_counts.items():
            str_value = str(value)
            
            if include_counts:
                col_template[str_value] = {
                    'suggested_mapping': str_value,  # Default to same value
                    'count': count
                }
            else:
                col_template[str_value] = str_value
        
        template[col] = col_template
        
        logger.info(f"Column {col}: found {len(col_template)} unique values")
    
    return template


def validate_dictionary_mappings(
    data: pd.DataFrame,
    dictionary: Dict[str, Dict[str, str]]
) -> Dict[str, Any]:
    """
    Validate dictionary mappings against actual data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
    dictionary : Dict[str, Dict[str, str]]
        Dictionary to validate
        
    Returns
    -------
    Dict[str, Any]
        Validation report with coverage and missing values
    """
    validate_dataframe(data)
    
    validation_report = {
        'columns': {},
        'summary': {}
    }
    
    for col, mappings in dictionary.items():
        if col not in data.columns:
            validation_report['columns'][col] = {
                'status': 'error',
                'message': 'Column not found in data'
            }
            continue
        
        # Get unique values in the column
        unique_values = set(data[col].dropna().astype(str).unique())
        
        # Get mapped values
        mapped_values = set(mappings.keys())
        
        # Calculate coverage
        covered_values = unique_values & mapped_values
        missing_values = unique_values - mapped_values
        extra_mappings = mapped_values - unique_values
        
        coverage_rate = len(covered_values) / len(unique_values) if unique_values else 0
        
        validation_report['columns'][col] = {
            'status': 'valid' if coverage_rate >= 0.9 else 'warning',
            'total_unique_values': len(unique_values),
            'mapped_values': len(covered_values),
            'missing_values': list(missing_values),
            'extra_mappings': list(extra_mappings),
            'coverage_rate': coverage_rate
        }
    
    # Overall summary
    total_columns = len(dictionary)
    valid_columns = sum(1 for col_data in validation_report['columns'].values() 
                       if col_data.get('status') == 'valid')
    
    validation_report['summary'] = {
        'total_columns': total_columns,
        'valid_columns': valid_columns,
        'warning_columns': total_columns - valid_columns
    }
    
    return validation_report