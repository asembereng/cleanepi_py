"""
Main data cleaning function that orchestrates all cleaning operations.

This is the primary entry point for the cleanepi package.
"""

from typing import Tuple, Optional, Union
import pandas as pd
from datetime import datetime
from loguru import logger

from .config import CleaningConfig
from .report import CleaningReport, OperationResult
from ..utils.validation import validate_dataframe, validate_config
from ..cleaning.standardize_columns import standardize_column_names
from ..cleaning.replace_missing import replace_missing_values
from ..cleaning.remove_constants import remove_constants
from ..cleaning.remove_duplicates import remove_duplicates
from ..cleaning.standardize_dates import standardize_dates
from ..cleaning.validate_subject_ids import check_subject_ids
from ..cleaning.convert_numeric import convert_to_numeric
from ..cleaning.dictionary_cleaning import clean_using_dictionary
from ..cleaning.date_sequence import check_date_sequence


def clean_data(
    data: pd.DataFrame,
    config: Optional[CleaningConfig] = None,
    **kwargs
) -> Tuple[pd.DataFrame, CleaningReport]:
    """
    Clean and standardize epidemiological data.
    
    This function performs comprehensive data cleaning operations including:
    - Column name standardization
    - Missing value replacement
    - Constant column removal
    - Duplicate removal
    - Date standardization
    - Subject ID validation
    - Numeric conversion
    - Dictionary-based cleaning
    - Date sequence validation
    
    Parameters
    ----------
    data : pd.DataFrame
        The input dataframe to clean
    config : CleaningConfig, optional
        Configuration object specifying cleaning operations.
        If None, uses default configuration.
    **kwargs
        Additional keyword arguments to override config values
        
    Returns
    -------
    Tuple[pd.DataFrame, CleaningReport]
        Cleaned dataframe and detailed cleaning report
        
    Raises
    ------
    ValueError
        If input data is invalid or configuration is incorrect
    TypeError
        If data is not a pandas DataFrame
        
    Examples
    --------
    >>> import pandas as pd
    >>> from cleanepi import clean_data, CleaningConfig
    >>> 
    >>> # Basic usage with default config
    >>> data = pd.DataFrame({'id': ['001', '002'], 'value': ['-99', '1']})
    >>> cleaned_data, report = clean_data(data)
    >>> 
    >>> # Advanced usage with custom config
    >>> config = CleaningConfig(
    ...     standardize_dates=DateConfig(target_columns=['date_col']),
    ...     replace_missing_values=MissingValueConfig(na_strings=['-99'])
    ... )
    >>> cleaned_data, report = clean_data(data, config)
    """
    
    # Input validation
    validate_dataframe(data)
    
    if config is None:
        config = CleaningConfig()
    
    # Override config with kwargs
    if kwargs:
        config_dict = config.dict()
        config_dict.update(kwargs)
        config = CleaningConfig(**config_dict)
    
    validate_config(config)
    
    # Initialize report
    report = CleaningReport(initial_shape=data.shape)
    current_data = data.copy()
    
    logger.info(f"Starting data cleaning with shape {data.shape}")
    
    try:
        # 1. Standardize column names
        if config.standardize_column_names:
            current_data, op_result = _standardize_column_names(
                current_data, config.standardize_column_names
            )
            report.add_operation(op_result)
        
        # 2. Replace missing values
        if config.replace_missing_values:
            current_data, op_result = _replace_missing_values(
                current_data, config.replace_missing_values
            )
            report.add_operation(op_result)
        
        # 3. Remove constant columns
        if config.remove_constants:
            current_data, op_result = _remove_constants(
                current_data, config.remove_constants
            )
            report.add_operation(op_result)
        
        # 4. Remove duplicates
        if config.remove_duplicates:
            current_data, op_result = _remove_duplicates(
                current_data, config.remove_duplicates
            )
            report.add_operation(op_result)
        
        # 5. Standardize dates
        if config.standardize_dates:
            current_data, op_result = _standardize_dates(
                current_data, config.standardize_dates
            )
            report.add_operation(op_result)
        
        # 6. Check subject IDs
        if config.standardize_subject_ids:
            current_data, op_result = _check_subject_ids(
                current_data, config.standardize_subject_ids
            )
            report.add_operation(op_result)
        
        # 7. Convert to numeric
        if config.to_numeric:
            current_data, op_result = _convert_to_numeric(
                current_data, config.to_numeric
            )
            report.add_operation(op_result)
        
        # 8. Dictionary-based cleaning
        if config.dictionary:
            current_data, op_result = _clean_using_dictionary(
                current_data, config.dictionary
            )
            report.add_operation(op_result)
        
        # 9. Check date sequence
        if config.check_date_sequence:
            current_data, op_result = _check_date_sequence(
                current_data, config.check_date_sequence
            )
            report.add_operation(op_result)
        
        # Finalize report
        report.finalize(current_data.shape)
        
        logger.info(f"Data cleaning completed. Final shape: {current_data.shape}")
        
        if config.verbose:
            report.print_summary()
        
        return current_data, report
        
    except Exception as e:
        logger.error(f"Data cleaning failed: {str(e)}")
        report.finalize(current_data.shape)
        if config.strict_validation:
            raise
        return current_data, report


def _standardize_column_names(
    data: pd.DataFrame, 
    config: Union[bool, dict]
) -> Tuple[pd.DataFrame, OperationResult]:
    """Wrapper for column name standardization."""
    start_time = datetime.now()
    initial_shape = data.shape
    
    try:
        if isinstance(config, bool) and config:
            # Use default settings
            cleaned_data = standardize_column_names(data)
        elif isinstance(config, dict):
            cleaned_data = standardize_column_names(data, **config)
        else:
            cleaned_data = data
            
        result = OperationResult(
            operation="standardize_column_names",
            timestamp=start_time,
            success=True,
            rows_before=initial_shape[0],
            rows_after=cleaned_data.shape[0],
            columns_before=initial_shape[1],
            columns_after=cleaned_data.shape[1],
            details={
                "original_columns": list(data.columns),
                "new_columns": list(cleaned_data.columns),
                "renamed_count": sum(1 for old, new in zip(data.columns, cleaned_data.columns) if old != new)
            }
        )
        
        return cleaned_data, result
        
    except Exception as e:
        result = OperationResult(
            operation="standardize_column_names",
            timestamp=start_time,
            success=False,
            rows_before=initial_shape[0],
            rows_after=initial_shape[0],
            columns_before=initial_shape[1],
            columns_after=initial_shape[1],
            errors=[str(e)]
        )
        return data, result


def _replace_missing_values(
    data: pd.DataFrame, 
    config
) -> Tuple[pd.DataFrame, OperationResult]:
    """Wrapper for missing value replacement."""
    start_time = datetime.now()
    initial_shape = data.shape
    
    try:
        cleaned_data = replace_missing_values(
            data,
            target_columns=config.target_columns,
            na_strings=config.na_strings,
            custom_na_by_column=config.custom_na_by_column
        )
        
        # Count replacements
        total_replacements = 0
        for col in cleaned_data.columns:
            if col in data.columns:
                replacements = (~data[col].isna() & cleaned_data[col].isna()).sum()
                total_replacements += replacements
        
        result = OperationResult(
            operation="replace_missing_values",
            timestamp=start_time,
            success=True,
            rows_before=initial_shape[0],
            rows_after=cleaned_data.shape[0],
            columns_before=initial_shape[1],
            columns_after=cleaned_data.shape[1],
            details={
                "total_replacements": total_replacements,
                "na_strings": config.na_strings
            }
        )
        
        return cleaned_data, result
        
    except Exception as e:
        result = OperationResult(
            operation="replace_missing_values",
            timestamp=start_time,
            success=False,
            rows_before=initial_shape[0],
            rows_after=initial_shape[0],
            columns_before=initial_shape[1],
            columns_after=initial_shape[1],
            errors=[str(e)]
        )
        return data, result


def _remove_constants(
    data: pd.DataFrame, 
    config
) -> Tuple[pd.DataFrame, OperationResult]:
    """Wrapper for constant removal."""
    start_time = datetime.now()
    initial_shape = data.shape
    
    try:
        cleaned_data = remove_constants(
            data,
            cutoff=config.cutoff,
            exclude_columns=config.exclude_columns
        )
        
        removed_columns = set(data.columns) - set(cleaned_data.columns)
        
        result = OperationResult(
            operation="remove_constants",
            timestamp=start_time,
            success=True,
            rows_before=initial_shape[0],
            rows_after=cleaned_data.shape[0],
            columns_before=initial_shape[1],
            columns_after=cleaned_data.shape[1],
            details={
                "removed_columns": list(removed_columns),
                "cutoff": config.cutoff
            }
        )
        
        return cleaned_data, result
        
    except Exception as e:
        result = OperationResult(
            operation="remove_constants",
            timestamp=start_time,
            success=False,
            rows_before=initial_shape[0],
            rows_after=initial_shape[0],
            columns_before=initial_shape[1],
            columns_after=initial_shape[1],
            errors=[str(e)]
        )
        return data, result


def _remove_duplicates(
    data: pd.DataFrame, 
    config
) -> Tuple[pd.DataFrame, OperationResult]:
    """Wrapper for duplicate removal."""
    start_time = datetime.now()
    initial_shape = data.shape
    
    try:
        cleaned_data = remove_duplicates(
            data,
            target_columns=config.target_columns or config.subset,
            keep=config.keep
        )
        
        duplicates_removed = initial_shape[0] - cleaned_data.shape[0]
        
        result = OperationResult(
            operation="remove_duplicates",
            timestamp=start_time,
            success=True,
            rows_before=initial_shape[0],
            rows_after=cleaned_data.shape[0],
            columns_before=initial_shape[1],
            columns_after=cleaned_data.shape[1],
            details={
                "duplicates_removed": duplicates_removed,
                "target_columns": config.target_columns or config.subset,
                "keep": config.keep
            }
        )
        
        return cleaned_data, result
        
    except Exception as e:
        result = OperationResult(
            operation="remove_duplicates",
            timestamp=start_time,
            success=False,
            rows_before=initial_shape[0],
            rows_after=initial_shape[0],
            columns_before=initial_shape[1],
            columns_after=initial_shape[1],
            errors=[str(e)]
        )
        return data, result


def _standardize_dates(
    data: pd.DataFrame, 
    config
) -> Tuple[pd.DataFrame, OperationResult]:
    """Wrapper for date standardization."""
    start_time = datetime.now()
    initial_shape = data.shape
    
    try:
        cleaned_data = standardize_dates(
            data,
            target_columns=config.target_columns,
            formats=config.formats,
            timeframe=config.timeframe,
            error_tolerance=config.error_tolerance,
            orders=config.orders
        )
        
        result = OperationResult(
            operation="standardize_dates",
            timestamp=start_time,
            success=True,
            rows_before=initial_shape[0],
            rows_after=cleaned_data.shape[0],
            columns_before=initial_shape[1],
            columns_after=cleaned_data.shape[1],
            details={
                "target_columns": config.target_columns,
                "error_tolerance": config.error_tolerance
            }
        )
        
        return cleaned_data, result
        
    except Exception as e:
        result = OperationResult(
            operation="standardize_dates",
            timestamp=start_time,
            success=False,
            rows_before=initial_shape[0],
            rows_after=initial_shape[0],
            columns_before=initial_shape[1],
            columns_after=initial_shape[1],
            errors=[str(e)]
        )
        return data, result


def _check_subject_ids(
    data: pd.DataFrame, 
    config
) -> Tuple[pd.DataFrame, OperationResult]:
    """Wrapper for subject ID checking."""
    start_time = datetime.now()
    initial_shape = data.shape
    
    try:
        cleaned_data = check_subject_ids(
            data,
            target_columns=config.target_columns,
            prefix=config.prefix,
            suffix=config.suffix,
            range=config.range,
            nchar=config.nchar,
            pattern=config.pattern
        )
        
        result = OperationResult(
            operation="check_subject_ids",
            timestamp=start_time,
            success=True,
            rows_before=initial_shape[0],
            rows_after=cleaned_data.shape[0],
            columns_before=initial_shape[1],
            columns_after=cleaned_data.shape[1],
            details={
                "target_columns": config.target_columns,
                "prefix": config.prefix,
                "suffix": config.suffix
            }
        )
        
        return cleaned_data, result
        
    except Exception as e:
        result = OperationResult(
            operation="check_subject_ids",
            timestamp=start_time,
            success=False,
            rows_before=initial_shape[0],
            rows_after=initial_shape[0],
            columns_before=initial_shape[1],
            columns_after=initial_shape[1],
            errors=[str(e)]
        )
        return data, result


def _convert_to_numeric(
    data: pd.DataFrame, 
    config
) -> Tuple[pd.DataFrame, OperationResult]:
    """Wrapper for numeric conversion."""
    start_time = datetime.now()
    initial_shape = data.shape
    
    try:
        cleaned_data = convert_to_numeric(
            data,
            target_columns=config.target_columns,
            lang=config.lang,
            errors=config.errors
        )
        
        result = OperationResult(
            operation="convert_to_numeric",
            timestamp=start_time,
            success=True,
            rows_before=initial_shape[0],
            rows_after=cleaned_data.shape[0],
            columns_before=initial_shape[1],
            columns_after=cleaned_data.shape[1],
            details={
                "target_columns": config.target_columns,
                "lang": config.lang
            }
        )
        
        return cleaned_data, result
        
    except Exception as e:
        result = OperationResult(
            operation="convert_to_numeric",
            timestamp=start_time,
            success=False,
            rows_before=initial_shape[0],
            rows_after=initial_shape[0],
            columns_before=initial_shape[1],
            columns_after=initial_shape[1],
            errors=[str(e)]
        )
        return data, result


def _clean_using_dictionary(
    data: pd.DataFrame, 
    dictionary: dict
) -> Tuple[pd.DataFrame, OperationResult]:
    """Wrapper for dictionary cleaning."""
    start_time = datetime.now()
    initial_shape = data.shape
    
    try:
        cleaned_data = clean_using_dictionary(data, dictionary)
        
        result = OperationResult(
            operation="clean_using_dictionary",
            timestamp=start_time,
            success=True,
            rows_before=initial_shape[0],
            rows_after=cleaned_data.shape[0],
            columns_before=initial_shape[1],
            columns_after=cleaned_data.shape[1],
            details={
                "dictionary_columns": list(dictionary.keys()),
                "total_mappings": sum(len(mappings) for mappings in dictionary.values())
            }
        )
        
        return cleaned_data, result
        
    except Exception as e:
        result = OperationResult(
            operation="clean_using_dictionary",
            timestamp=start_time,
            success=False,
            rows_before=initial_shape[0],
            rows_after=initial_shape[0],
            columns_before=initial_shape[1],
            columns_after=initial_shape[1],
            errors=[str(e)]
        )
        return data, result


def _check_date_sequence(
    data: pd.DataFrame, 
    target_columns: list
) -> Tuple[pd.DataFrame, OperationResult]:
    """Wrapper for date sequence checking."""
    start_time = datetime.now()
    initial_shape = data.shape
    
    try:
        cleaned_data = check_date_sequence(data, target_columns)
        
        result = OperationResult(
            operation="check_date_sequence",
            timestamp=start_time,
            success=True,
            rows_before=initial_shape[0],
            rows_after=cleaned_data.shape[0],
            columns_before=initial_shape[1],
            columns_after=cleaned_data.shape[1],
            details={
                "target_columns": target_columns
            }
        )
        
        return cleaned_data, result
        
    except Exception as e:
        result = OperationResult(
            operation="check_date_sequence",
            timestamp=start_time,
            success=False,
            rows_before=initial_shape[0],
            rows_after=initial_shape[0],
            columns_before=initial_shape[1],
            columns_after=initial_shape[1],
            errors=[str(e)]
        )
        return data, result