"""Date sequence validation functionality."""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger

from ..utils.validation import validate_dataframe, validate_columns_exist


def check_date_sequence(
    data: pd.DataFrame,
    target_columns: List[str],
    tolerance_days: int = 0,
    allow_equal: bool = True,
    subject_id_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Check date sequence validity and logical ordering.
    
    This function validates date sequences by:
    - Checking chronological order of dates
    - Identifying impossible date combinations
    - Flagging dates outside acceptable ranges
    - Validating per-subject date sequences
    - Detecting duplicate dates when not allowed
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
    target_columns : List[str]
        Date columns to check in chronological order (earliest to latest expected)
    tolerance_days : int, default 0
        Number of days tolerance for date order (negative values allow reverse order)
    allow_equal : bool, default True
        Whether to allow equal dates in sequence
    subject_id_column : str, optional
        Column containing subject IDs for per-subject validation
        
    Returns
    -------
    pd.DataFrame
        DataFrame with validation results added as new columns
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'patient_id': ['P001', 'P002', 'P003'],
    ...     'birth_date': ['1990-01-15', '1985-03-20', '1995-07-10'],
    ...     'admission_date': ['2023-01-10', '2023-02-15', '2023-01-05'],
    ...     'discharge_date': ['2023-01-20', '2023-02-10', '2023-01-15']
    ... })
    >>> result = check_date_sequence(
    ...     df, 
    ...     ['birth_date', 'admission_date', 'discharge_date'],
    ...     subject_id_column='patient_id'
    ... )
    >>> # Adds validation columns showing sequence validity
    """
    validate_dataframe(data)
    validate_columns_exist(data, target_columns, "check_date_sequence")
    
    if len(target_columns) < 2:
        logger.warning("Need at least 2 date columns for sequence validation")
        return data.copy()
    
    if subject_id_column and subject_id_column not in data.columns:
        logger.warning(f"Subject ID column '{subject_id_column}' not found, ignoring")
        subject_id_column = None
    
    # Make a copy to avoid modifying original
    result = data.copy()
    
    logger.info(f"Checking date sequence for {len(target_columns)} columns")
    
    # Convert date columns to datetime if needed
    date_columns = {}
    for col in target_columns:
        if pd.api.types.is_datetime64_any_dtype(result[col]):
            date_columns[col] = result[col]
        else:
            try:
                date_columns[col] = pd.to_datetime(result[col], errors='coerce')
                logger.info(f"Converted column {col} to datetime")
            except Exception as e:
                logger.error(f"Failed to convert column {col} to datetime: {e}")
                return result
    
    # Perform validation
    if subject_id_column:
        validation_results = _validate_sequences_by_subject(
            date_columns, result[subject_id_column], tolerance_days, allow_equal
        )
    else:
        validation_results = _validate_sequences_global(
            date_columns, tolerance_days, allow_equal
        )
    
    # Add validation columns to result
    for col_name, col_data in validation_results.items():
        result[col_name] = col_data
    
    # Log summary
    valid_sequences = result['date_sequence_valid'].sum()
    total_sequences = len(result)
    logger.info(
        f"Date sequence validation: {valid_sequences}/{total_sequences} valid sequences "
        f"({(valid_sequences/total_sequences)*100:.1f}%)"
    )
    
    # Log common issues
    if 'date_sequence_issues' in result.columns:
        issue_counts = {}
        for issues_str in result['date_sequence_issues']:
            if issues_str and str(issues_str).strip():
                issues = str(issues_str).split(', ')
                for issue in issues:
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        if issue_counts:
            logger.info("Common date sequence issues:")
            for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {issue}: {count} occurrences")
    
    return result


def _validate_sequences_by_subject(
    date_columns: Dict[str, pd.Series],
    subject_ids: pd.Series,
    tolerance_days: int,
    allow_equal: bool
) -> Dict[str, pd.Series]:
    """
    Validate date sequences grouped by subject ID.
    
    Parameters
    ----------
    date_columns : Dict[str, pd.Series]
        Dictionary of date columns
    subject_ids : pd.Series
        Subject ID column
    tolerance_days : int
        Tolerance for date order
    allow_equal : bool
        Whether to allow equal dates
        
    Returns
    -------
    Dict[str, pd.Series]
        Validation result columns
    """
    n_rows = len(subject_ids)
    valid_series = pd.Series(True, index=subject_ids.index)
    issues_series = pd.Series('', index=subject_ids.index)
    
    # Group by subject ID
    unique_subjects = subject_ids.dropna().unique()
    column_names = list(date_columns.keys())
    
    for subject_id in unique_subjects:
        subject_mask = subject_ids == subject_id
        subject_indices = subject_ids[subject_mask].index
        
        # Get dates for this subject
        subject_dates = {}
        for col_name, col_data in date_columns.items():
            subject_dates[col_name] = col_data[subject_mask]
        
        # Validate sequence for this subject
        for idx in subject_indices:
            row_dates = [subject_dates[col].loc[idx] for col in column_names]
            
            # Skip if any dates are missing
            if any(pd.isna(date) for date in row_dates):
                issues_series.loc[idx] = 'missing_dates'
                valid_series.loc[idx] = False
                continue
            
            # Check sequence
            sequence_valid, sequence_issues = _check_single_sequence(
                row_dates, column_names, tolerance_days, allow_equal
            )
            
            valid_series.loc[idx] = sequence_valid
            if sequence_issues:
                issues_series.loc[idx] = ', '.join(sequence_issues)
    
    return {
        'date_sequence_valid': valid_series,
        'date_sequence_issues': issues_series
    }


def _validate_sequences_global(
    date_columns: Dict[str, pd.Series],
    tolerance_days: int,
    allow_equal: bool
) -> Dict[str, pd.Series]:
    """
    Validate date sequences globally (row by row).
    
    Parameters
    ----------
    date_columns : Dict[str, pd.Series]
        Dictionary of date columns
    tolerance_days : int
        Tolerance for date order
    allow_equal : bool
        Whether to allow equal dates
        
    Returns
    -------
    Dict[str, pd.Series]
        Validation result columns
    """
    column_names = list(date_columns.keys())
    first_column = date_columns[column_names[0]]
    
    valid_series = pd.Series(True, index=first_column.index)
    issues_series = pd.Series('', index=first_column.index)
    
    for idx in first_column.index:
        row_dates = [date_columns[col].loc[idx] for col in column_names]
        
        # Skip if any dates are missing
        if any(pd.isna(date) for date in row_dates):
            issues_series.loc[idx] = 'missing_dates'
            valid_series.loc[idx] = False
            continue
        
        # Check sequence
        sequence_valid, sequence_issues = _check_single_sequence(
            row_dates, column_names, tolerance_days, allow_equal
        )
        
        valid_series.loc[idx] = sequence_valid
        if sequence_issues:
            issues_series.loc[idx] = ', '.join(sequence_issues)
    
    return {
        'date_sequence_valid': valid_series,
        'date_sequence_issues': issues_series
    }


def _check_single_sequence(
    dates: List[pd.Timestamp],
    column_names: List[str],
    tolerance_days: int,
    allow_equal: bool
) -> Tuple[bool, List[str]]:
    """
    Check a single date sequence for validity.
    
    Parameters
    ----------
    dates : List[pd.Timestamp]
        List of dates in expected chronological order
    column_names : List[str]
        Names of the date columns
    tolerance_days : int
        Tolerance for date order
    allow_equal : bool
        Whether to allow equal dates
        
    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_issues)
    """
    issues = []
    is_valid = True
    
    # Check each consecutive pair
    for i in range(len(dates) - 1):
        date1 = dates[i]
        date2 = dates[i + 1]
        col1 = column_names[i]
        col2 = column_names[i + 1]
        
        # Calculate difference in days
        date_diff = (date2 - date1).days
        
        # Check if dates are in correct order
        if date_diff < -tolerance_days:
            issues.append(f"{col2}_before_{col1}")
            is_valid = False
        elif not allow_equal and date_diff == 0:
            issues.append(f"{col1}_equals_{col2}")
            is_valid = False
        elif abs(date_diff) > tolerance_days and tolerance_days >= 0:
            # Check for suspiciously large gaps (optional validation)
            if date_diff > 365 * 150:  # More than 150 years
                issues.append(f"large_gap_{col1}_to_{col2}")
    
    # Additional logical checks
    if len(dates) >= 3:
        # Check for birth date in future (if first date might be birth date)
        first_date = dates[0]
        last_date = dates[-1]
        now = pd.Timestamp.now()
        
        if first_date > now:
            issues.append("future_birth_date")
            is_valid = False
        
        # Check for extremely old birth dates
        if first_date < pd.Timestamp('1900-01-01'):
            issues.append("ancient_birth_date")
            is_valid = False
        
        # Check for impossible lifespans
        if (last_date - first_date).days > 365 * 120:  # More than 120 years
            issues.append("impossible_lifespan")
            is_valid = False
    
    return is_valid, issues


def generate_date_sequence_report(
    data: pd.DataFrame,
    target_columns: List[str]
) -> Dict[str, Any]:
    """
    Generate a comprehensive report on date sequence validation.
    
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
        'summary': {},
        'column_analysis': {},
        'issue_breakdown': {},
        'recommendations': []
    }
    
    if 'date_sequence_valid' not in data.columns:
        logger.warning("No date sequence validation results found")
        return report
    
    # Overall summary
    total_rows = len(data)
    valid_sequences = data['date_sequence_valid'].sum()
    invalid_sequences = total_rows - valid_sequences
    
    report['summary'] = {
        'total_sequences': total_rows,
        'valid_sequences': valid_sequences,
        'invalid_sequences': invalid_sequences,
        'valid_percentage': (valid_sequences / total_rows) * 100 if total_rows > 0 else 0
    }
    
    # Column-specific analysis
    for col in target_columns:
        if col in data.columns:
            non_null_count = data[col].notna().sum()
            null_count = data[col].isna().sum()
            
            # Date range analysis
            date_col = pd.to_datetime(data[col], errors='coerce')
            min_date = date_col.min()
            max_date = date_col.max()
            
            report['column_analysis'][col] = {
                'non_null_count': non_null_count,
                'null_count': null_count,
                'min_date': str(min_date) if pd.notna(min_date) else None,
                'max_date': str(max_date) if pd.notna(max_date) else None,
                'date_range_days': (max_date - min_date).days if pd.notna(min_date) and pd.notna(max_date) else None
            }
    
    # Issue breakdown
    if 'date_sequence_issues' in data.columns:
        issue_counts = {}
        for issues_str in data['date_sequence_issues']:
            if issues_str and str(issues_str).strip():
                issues = str(issues_str).split(', ')
                for issue in issues:
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        report['issue_breakdown'] = issue_counts
    
    # Generate recommendations
    recommendations = []
    
    if report['summary']['valid_percentage'] < 90:
        recommendations.append("Consider reviewing date entry procedures - less than 90% of sequences are valid")
    
    if 'missing_dates' in report['issue_breakdown']:
        missing_count = report['issue_breakdown']['missing_dates']
        recommendations.append(f"Address {missing_count} records with missing dates")
    
    if any('before' in issue for issue in report['issue_breakdown'].keys()):
        recommendations.append("Review records with reversed date sequences")
    
    if any('future' in issue for issue in report['issue_breakdown'].keys()):
        recommendations.append("Check for future dates that may be data entry errors")
    
    report['recommendations'] = recommendations
    
    return report


def detect_date_outliers(
    data: pd.DataFrame,
    date_columns: List[str],
    method: str = "iqr",
    factor: float = 1.5
) -> pd.DataFrame:
    """
    Detect outlier dates that may indicate data entry errors.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
    date_columns : List[str]
        Date columns to analyze
    method : str, default "iqr"
        Method for outlier detection: "iqr", "std", "percentile"
    factor : float, default 1.5
        Multiplier for outlier detection threshold
        
    Returns
    -------
    pd.DataFrame
        DataFrame with outlier flags added
    """
    validate_dataframe(data)
    validate_columns_exist(data, date_columns, "detect_date_outliers")
    
    result = data.copy()
    
    logger.info(f"Detecting date outliers in {len(date_columns)} columns using {method} method")
    
    for col in date_columns:
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(result[col]):
            date_series = pd.to_datetime(result[col], errors='coerce')
        else:
            date_series = result[col]
        
        # Convert to numeric (days since epoch) for outlier detection
        numeric_dates = date_series.astype('int64') / (24 * 3600 * 1e9)  # Convert to days
        
        if method == "iqr":
            Q1 = numeric_dates.quantile(0.25)
            Q3 = numeric_dates.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            outliers = (numeric_dates < lower_bound) | (numeric_dates > upper_bound)
            
        elif method == "std":
            mean_date = numeric_dates.mean()
            std_date = numeric_dates.std()
            
            lower_bound = mean_date - factor * std_date
            upper_bound = mean_date + factor * std_date
            
            outliers = (numeric_dates < lower_bound) | (numeric_dates > upper_bound)
            
        elif method == "percentile":
            lower_percentile = (50 - factor * 10) / 100  # Adjust percentiles based on factor
            upper_percentile = (50 + factor * 10) / 100
            
            lower_bound = numeric_dates.quantile(max(0, lower_percentile))
            upper_bound = numeric_dates.quantile(min(1, upper_percentile))
            
            outliers = (numeric_dates < lower_bound) | (numeric_dates > upper_bound)
            
        else:
            logger.warning(f"Unknown outlier detection method: {method}")
            continue
        
        # Add outlier flag column
        outlier_col = f"{col}_outlier"
        result[outlier_col] = outliers.fillna(False)
        
        outlier_count = outliers.sum()
        total_dates = date_series.notna().sum()
        
        logger.info(
            f"Column {col}: {outlier_count}/{total_dates} outlier dates detected "
            f"({(outlier_count/total_dates)*100:.1f}%)"
        )
    
    return result