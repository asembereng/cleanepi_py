"""Subject ID validation functionality."""

from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
import re
from loguru import logger

from ..utils.validation import validate_dataframe, validate_columns_exist


def check_subject_ids(
    data: pd.DataFrame,
    target_columns: List[str],
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    range: Optional[Tuple[int, int]] = None,
    nchar: Optional[int] = None,
    pattern: Optional[str] = None
) -> pd.DataFrame:
    """
    Check and validate subject IDs against specified criteria.
    
    This function validates subject IDs by checking:
    - Prefix/suffix patterns
    - Numeric ranges
    - Character length requirements
    - Custom regex patterns
    - Duplicate IDs across columns
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
    target_columns : List[str]
        Column names containing subject IDs to validate
    prefix : str, optional
        Expected prefix for subject IDs
    suffix : str, optional
        Expected suffix for subject IDs
    range : Tuple[int, int], optional
        Valid numeric range for ID numbers as (min, max)
    nchar : int, optional
        Expected total character length
    pattern : str, optional
        Custom regex pattern for validation
        
    Returns
    -------
    pd.DataFrame
        DataFrame with validation results added as new columns
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'patient_id': ['P001', 'P002', 'P999', 'X123'],
    ...     'study_id': ['S001', 'S002', 'S003', 'S004']
    ... })
    >>> result = check_subject_ids(
    ...     df, 
    ...     target_columns=['patient_id'],
    ...     prefix='P',
    ...     nchar=4
    ... )
    >>> # Adds validation columns showing which IDs are valid/invalid
    """
    validate_dataframe(data)
    validate_columns_exist(data, target_columns, "check_subject_ids")
    
    # Make a copy to avoid modifying original
    result = data.copy()
    
    logger.info(f"Validating subject IDs in {len(target_columns)} columns")
    
    validation_summary = {}
    
    for col in target_columns:
        logger.info(f"Validating column: {col}")
        
        # Get non-null values
        ids = result[col].dropna().astype(str)
        
        if len(ids) == 0:
            logger.warning(f"Column {col} has no non-null values")
            continue
        
        # Initialize validation results
        validation_results = _validate_ids(
            ids, prefix=prefix, suffix=suffix, 
            range=range, nchar=nchar, pattern=pattern
        )
        
        # Add validation columns to result
        valid_col = f"{col}_valid"
        issues_col = f"{col}_issues"
        
        # Create validation series for all rows
        valid_series = pd.Series(True, index=result.index)
        issues_series = pd.Series('', index=result.index)
        
        # Update validation results for non-null values
        for idx, id_val in ids.items():
            if id_val in validation_results:
                valid_series.loc[idx] = validation_results[id_val]['valid']
                issues_series.loc[idx] = ', '.join(validation_results[id_val]['issues'])
        
        # Set null values as invalid
        null_mask = result[col].isna()
        valid_series.loc[null_mask] = False
        issues_series.loc[null_mask] = 'missing_value'
        
        result[valid_col] = valid_series
        result[issues_col] = issues_series
        
        # Generate summary
        total_ids = len(result)
        valid_ids = valid_series.sum()
        invalid_ids = total_ids - valid_ids
        
        validation_summary[col] = {
            'total': total_ids,
            'valid': valid_ids,
            'invalid': invalid_ids,
            'valid_percentage': (valid_ids / total_ids) * 100 if total_ids > 0 else 0
        }
        
        logger.info(
            f"Column {col}: {valid_ids}/{total_ids} valid IDs "
            f"({validation_summary[col]['valid_percentage']:.1f}%)"
        )
    
    # Check for duplicate IDs across columns
    if len(target_columns) > 1:
        duplicate_summary = _check_cross_column_duplicates(result, target_columns)
        if duplicate_summary:
            logger.warning("Found duplicate IDs across columns:")
            for dup_id, columns in duplicate_summary.items():
                logger.warning(f"  ID '{dup_id}' appears in columns: {columns}")
    
    # Log overall summary
    total_all = sum(s['total'] for s in validation_summary.values())
    valid_all = sum(s['valid'] for s in validation_summary.values())
    if total_all > 0:
        logger.info(
            f"Overall validation: {valid_all}/{total_all} valid IDs "
            f"({(valid_all/total_all)*100:.1f}%)"
        )
    
    return result


def _validate_ids(
    ids: pd.Series,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    range: Optional[Tuple[int, int]] = None,
    nchar: Optional[int] = None,
    pattern: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Validate individual IDs against criteria.
    
    Parameters
    ----------
    ids : pd.Series
        Series of ID strings to validate
    prefix : str, optional
        Expected prefix
    suffix : str, optional
        Expected suffix
    range : Tuple[int, int], optional
        Valid numeric range for ID numbers
    nchar : int, optional
        Expected character length
    pattern : str, optional
        Custom regex pattern
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Validation results for each ID
    """
    results = {}
    
    for id_val in ids.unique():
        if pd.isna(id_val) or str(id_val).strip() == '':
            continue
            
        id_str = str(id_val).strip()
        issues = []
        valid = True
        
        # Check character length
        if nchar is not None and len(id_str) != nchar:
            issues.append(f"length_mismatch_expected_{nchar}_got_{len(id_str)}")
            valid = False
        
        # Check prefix
        if prefix is not None and not id_str.startswith(prefix):
            issues.append(f"missing_prefix_{prefix}")
            valid = False
        
        # Check suffix
        if suffix is not None and not id_str.endswith(suffix):
            issues.append(f"missing_suffix_{suffix}")
            valid = False
        
        # Check custom pattern
        if pattern is not None:
            try:
                if not re.match(pattern, id_str):
                    issues.append(f"pattern_mismatch")
                    valid = False
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
                issues.append("invalid_pattern")
                valid = False
        
        # Check numeric range (extract numeric part)
        if range is not None:
            numeric_part = _extract_numeric_part(id_str, prefix, suffix)
            if numeric_part is not None:
                try:
                    num_val = int(numeric_part)
                    min_val, max_val = range
                    if not (min_val <= num_val <= max_val):
                        issues.append(f"range_violation_expected_{min_val}_to_{max_val}_got_{num_val}")
                        valid = False
                except ValueError:
                    issues.append("non_numeric_id_part")
                    valid = False
            else:
                issues.append("no_numeric_part_found")
                valid = False
        
        results[id_str] = {
            'valid': valid,
            'issues': issues
        }
    
    return results


def _extract_numeric_part(
    id_str: str, 
    prefix: Optional[str] = None, 
    suffix: Optional[str] = None
) -> Optional[str]:
    """
    Extract numeric part from ID string.
    
    Parameters
    ----------
    id_str : str
        ID string
    prefix : str, optional
        Known prefix to remove
    suffix : str, optional
        Known suffix to remove
        
    Returns
    -------
    str or None
        Numeric part of the ID, or None if not found
    """
    working_str = id_str
    
    # Remove known prefix
    if prefix and working_str.startswith(prefix):
        working_str = working_str[len(prefix):]
    
    # Remove known suffix
    if suffix and working_str.endswith(suffix):
        working_str = working_str[:-len(suffix)]
    
    # Extract numeric part using regex
    numeric_match = re.search(r'\d+', working_str)
    if numeric_match:
        return numeric_match.group()
    
    return None


def _check_cross_column_duplicates(
    data: pd.DataFrame, 
    target_columns: List[str]
) -> Dict[str, List[str]]:
    """
    Check for duplicate IDs across multiple columns.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing ID columns
    target_columns : List[str]
        Columns to check for duplicates
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping duplicate ID values to lists of columns they appear in
    """
    id_to_columns = {}
    
    for col in target_columns:
        ids = data[col].dropna().astype(str)
        for id_val in ids:
            if id_val not in id_to_columns:
                id_to_columns[id_val] = []
            if col not in id_to_columns[id_val]:
                id_to_columns[id_val].append(col)
    
    # Find IDs that appear in multiple columns
    duplicates = {
        id_val: columns 
        for id_val, columns in id_to_columns.items() 
        if len(columns) > 1
    }
    
    return duplicates


def generate_subject_id_report(data: pd.DataFrame, target_columns: List[str]) -> Dict[str, Any]:
    """
    Generate a comprehensive report on subject ID validation results.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with validation results
    target_columns : List[str]
        Original target columns
        
    Returns
    -------
    Dict[str, Any]
        Comprehensive validation report
    """
    report = {
        'columns': {},
        'summary': {},
        'issues': []
    }
    
    for col in target_columns:
        valid_col = f"{col}_valid"
        issues_col = f"{col}_issues"
        
        if valid_col in data.columns and issues_col in data.columns:
            total = len(data)
            valid = data[valid_col].sum()
            invalid = total - valid
            
            # Count issue types
            issue_counts = {}
            for issues_str in data[issues_col]:
                if issues_str and str(issues_str).strip():
                    issues = str(issues_str).split(', ')
                    for issue in issues:
                        issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
            report['columns'][col] = {
                'total_ids': total,
                'valid_ids': valid,
                'invalid_ids': invalid,
                'valid_percentage': (valid / total) * 100 if total > 0 else 0,
                'issue_breakdown': issue_counts
            }
    
    # Overall summary
    total_all = sum(col_data['total_ids'] for col_data in report['columns'].values())
    valid_all = sum(col_data['valid_ids'] for col_data in report['columns'].values())
    
    report['summary'] = {
        'total_ids': total_all,
        'valid_ids': valid_all,
        'invalid_ids': total_all - valid_all,
        'valid_percentage': (valid_all / total_all) * 100 if total_all > 0 else 0,
        'columns_processed': len(target_columns)
    }
    
    return report