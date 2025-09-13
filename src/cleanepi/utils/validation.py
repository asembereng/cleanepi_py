"""
Validation utilities for input data and configuration with security and performance checks.
"""

import os
import sys
import re
from typing import Any, List, Optional, Union
import pandas as pd
from loguru import logger

from ..core.config import CleaningConfig


def validate_dataframe(
    data: Any, 
    min_rows: int = 1, 
    min_cols: int = 1,
    max_memory_mb: Optional[float] = None
) -> None:
    """
    Validate that input is a proper pandas DataFrame with security and performance checks.
    
    Parameters
    ----------
    data : Any
        Input to validate
    min_rows : int
        Minimum required rows
    min_cols : int  
        Minimum required columns
    max_memory_mb : float, optional
        Maximum allowed memory usage in megabytes
        
    Raises
    ------
    TypeError
        If data is not a pandas DataFrame
    ValueError
        If DataFrame doesn't meet minimum requirements or security constraints
    MemoryError
        If DataFrame exceeds memory limits
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(data)}")
    
    if data.empty:
        raise ValueError("DataFrame is empty")
    
    if len(data) < min_rows:
        raise ValueError(f"DataFrame has {len(data)} rows, minimum {min_rows} required")
    
    if len(data.columns) < min_cols:
        raise ValueError(f"DataFrame has {len(data.columns)} columns, minimum {min_cols} required")
    
    # Memory usage validation
    memory_usage_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
    
    if max_memory_mb and memory_usage_mb > max_memory_mb:
        raise MemoryError(
            f"DataFrame memory usage ({memory_usage_mb:.1f} MB) exceeds "
            f"limit ({max_memory_mb} MB)"
        )
    
    # Security validation - check for suspicious column names
    _validate_column_names_security(data.columns)
    
    logger.debug(f"DataFrame validation passed: shape {data.shape}, memory {memory_usage_mb:.1f} MB")


def _validate_column_names_security(columns: pd.Index) -> None:
    """
    Validate column names for security issues.
    
    Parameters
    ----------
    columns : pd.Index
        DataFrame column names
        
    Raises
    ------
    ValueError
        If column names contain security risks
    """
    suspicious_patterns = [
        r'__.*__',  # Python dunder methods
        r'eval\s*\(',  # eval function calls
        r'exec\s*\(',  # exec function calls
        r'import\s+',  # import statements
        r'subprocess',  # subprocess calls
        r'os\.',  # os module calls
        r'sys\.',  # sys module calls
    ]
    
    for col in columns:
        col_str = str(col).lower()
        for pattern in suspicious_patterns:
            if re.search(pattern, col_str):
                logger.warning(f"Potentially suspicious column name detected: {col}")
                # Don't raise error, just warn - column names are usually safe


def validate_config(config: CleaningConfig) -> None:
    """
    Validate cleaning configuration with enhanced security checks.
    
    Parameters
    ----------
    config : CleaningConfig
        Configuration to validate
        
    Raises
    ------
    ValueError
        If configuration contains invalid or unsafe values
    """
    if not isinstance(config, CleaningConfig):
        raise TypeError(f"Expected CleaningConfig, got {type(config)}")
    
    # Validate memory limits
    if config.max_memory_usage:
        _validate_memory_limit(config.max_memory_usage)
    
    # Validate regex patterns in subject ID config
    if config.standardize_subject_ids and config.standardize_subject_ids.pattern:
        _validate_regex_pattern(config.standardize_subject_ids.pattern)
    
    # Validate dictionary mappings for code injection
    if config.dictionary:
        _validate_dictionary_security(config.dictionary)
    
    logger.debug("Configuration validation passed")


def _validate_memory_limit(memory_limit: str) -> None:
    """
    Validate memory limit string format.
    
    Parameters
    ----------
    memory_limit : str
        Memory limit string (e.g., "1GB", "500MB")
        
    Raises
    ------
    ValueError
        If memory limit format is invalid
    """
    pattern = r'^\d+(\.\d+)?(GB|MB|KB)$'
    if not re.match(pattern, memory_limit.upper()):
        raise ValueError(f"Invalid memory limit format: {memory_limit}")


def _validate_regex_pattern(pattern: str) -> None:
    """
    Validate regex pattern for security and correctness.
    
    Parameters
    ----------
    pattern : str
        Regex pattern to validate
        
    Raises
    ------
    ValueError
        If regex pattern is invalid or potentially unsafe
    """
    try:
        re.compile(pattern)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
    
    # Check for potentially dangerous regex patterns
    dangerous_patterns = [
        r'\$\{',  # Variable substitution
        r'eval\s*\(',  # eval calls
        r'exec\s*\(',  # exec calls
    ]
    
    for dangerous in dangerous_patterns:
        if re.search(dangerous, pattern, re.IGNORECASE):
            raise ValueError(f"Potentially unsafe regex pattern: {pattern}")


def _validate_dictionary_security(dictionary: dict) -> None:
    """
    Validate dictionary mappings for security issues.
    
    Parameters
    ----------
    dictionary : dict
        Dictionary mappings to validate
        
    Raises
    ------
    ValueError
        If dictionary contains potentially unsafe values
    """
    def check_value_security(value: str) -> None:
        """Check individual value for security issues."""
        if not isinstance(value, str):
            return
        
        # Check for code injection patterns
        suspicious_patterns = [
            r'<script',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'eval\s*\(',  # eval calls
            r'exec\s*\(',  # exec calls
            r'__import__',  # Python imports
            r'subprocess',  # subprocess calls
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"Potentially suspicious dictionary value: {value}")
    
    for column, mappings in dictionary.items():
        if not isinstance(mappings, dict):
            continue
        
        for old_val, new_val in mappings.items():
            check_value_security(str(old_val))
            check_value_security(str(new_val))


def validate_columns_exist(
    data: pd.DataFrame, 
    columns: List[str], 
    operation_name: str = "operation"
) -> None:
    """
    Validate that specified columns exist in the DataFrame.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
    columns : List[str]
        Required column names
    operation_name : str
        Name of operation for error messages
        
    Raises
    ------
    ValueError
        If any required columns are missing
    """
    missing_columns = [col for col in columns if col not in data.columns]
    
    if missing_columns:
        raise ValueError(
            f"{operation_name}: Missing columns {missing_columns}. "
            f"Available columns: {list(data.columns)}"
        )


def validate_file_safety(file_path: str, max_size_mb: float = 100) -> None:
    """
    Validate file for security and size constraints.
    
    Parameters
    ----------
    file_path : str
        Path to file
    max_size_mb : float
        Maximum allowed file size in megabytes
        
    Raises
    ------
    ValueError
        If file is unsafe or too large
    FileNotFoundError
        If file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Check for path traversal attacks
    abs_path = os.path.abspath(file_path)
    if '..' in file_path or abs_path != os.path.normpath(abs_path):
        raise ValueError(f"Potentially unsafe file path: {file_path}")
    
    # Check file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise ValueError(
            f"File size ({file_size_mb:.1f} MB) exceeds limit ({max_size_mb} MB)"
        )
    
    logger.debug(f"File validation passed: {file_path} ({file_size_mb:.1f} MB)")


def get_memory_usage() -> dict:
    """
    Get current memory usage statistics.
    
    Returns
    -------
    dict
        Memory usage information
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size
            'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / (1024 * 1024)
        }
    except ImportError:
        # Fallback if psutil not available
        return {
            'rss_mb': 0,
            'vms_mb': 0, 
            'percent': 0,
            'available_mb': 0
        }


def check_resource_limits(max_memory_mb: Optional[float] = None) -> bool:
    """
    Check if current resource usage is within limits.
    
    Parameters
    ----------
    max_memory_mb : float, optional
        Maximum allowed memory usage in MB
        
    Returns
    -------
    bool
        True if within limits, False otherwise
    """
    if max_memory_mb is None:
        return True
    
    memory_stats = get_memory_usage()
    current_memory = memory_stats['rss_mb']
    
    if current_memory > max_memory_mb:
        logger.warning(
            f"Memory usage ({current_memory:.1f} MB) exceeds limit ({max_memory_mb} MB)"
        )
        return False
    
    return True


def sanitize_input_string(input_str: str, max_length: int = 1000) -> str:
    """
    Sanitize input string for security.
    
    Parameters
    ----------
    input_str : str
        Input string to sanitize
    max_length : int
        Maximum allowed string length
        
    Returns
    -------
    str
        Sanitized string
        
    Raises
    ------
    ValueError
        If input is too long or contains dangerous patterns
    """
    if len(input_str) > max_length:
        raise ValueError(f"Input string too long: {len(input_str)} > {max_length}")
    
    # Remove potentially dangerous characters/patterns
    dangerous_patterns = [
        r'<script.*?>.*?</script>',  # Script tags
        r'javascript:',  # JavaScript URLs
        r'data:text/html',  # Data URLs
        r'vbscript:',  # VBScript
    ]
    
    sanitized = input_str
    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
    
    return sanitized.strip()


def validate_config(config) -> None:
    """
    Validate cleaning configuration.
    
    Parameters
    ----------
    config : CleaningConfig
        Configuration to validate
        
    Raises
    ------
    ValueError
        If configuration is invalid
    """
    if not isinstance(config, CleaningConfig):
        raise TypeError(f"Expected CleaningConfig, got {type(config)}")
    
    # Validate individual configs
    if config.standardize_subject_ids and not config.standardize_subject_ids.target_columns:
        raise ValueError("standardize_subject_ids requires target_columns")
    
    if config.to_numeric and not config.to_numeric.target_columns:
        raise ValueError("to_numeric requires target_columns")
    
    logger.debug("Configuration validation passed")


def validate_columns_exist(data: pd.DataFrame, columns: List[str], operation: str = "") -> None:
    """
    Validate that specified columns exist in DataFrame.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to check
    columns : List[str]
        Column names to validate
    operation : str
        Name of operation for error messages
        
    Raises
    ------
    ValueError
        If any columns don't exist
    """
    missing_columns = [col for col in columns if col not in data.columns]
    if missing_columns:
        raise ValueError(
            f"{operation}: Columns {missing_columns} not found in DataFrame. "
            f"Available columns: {list(data.columns)}"
        )


def validate_column_types(
    data: pd.DataFrame, 
    columns: List[str], 
    expected_types: List[type],
    operation: str = ""
) -> None:
    """
    Validate that columns have expected data types.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to check
    columns : List[str]
        Column names to validate
    expected_types : List[type]
        Expected data types
    operation : str
        Name of operation for error messages
        
    Raises
    ------
    ValueError
        If column types don't match expectations
    """
    validate_columns_exist(data, columns, operation)
    
    for col in columns:
        col_type = data[col].dtype
        if not any(pd.api.types.is_dtype_equal(col_type, expected_type) for expected_type in expected_types):
            raise ValueError(
                f"{operation}: Column '{col}' has type {col_type}, "
                f"expected one of {expected_types}"
            )


def check_memory_usage(data: pd.DataFrame, max_memory: Optional[str] = None) -> None:
    """
    Check if DataFrame memory usage is within limits.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to check
    max_memory : str, optional
        Maximum memory limit (e.g., '1GB', '500MB')
        
    Raises
    ------
    MemoryError
        If DataFrame exceeds memory limit
    """
    if max_memory is None:
        return
    
    # Parse memory limit
    memory_bytes = _parse_memory_string(max_memory)
    
    # Get DataFrame memory usage
    df_memory = data.memory_usage(deep=True).sum()
    
    if df_memory > memory_bytes:
        raise MemoryError(
            f"DataFrame memory usage ({df_memory / 1024**2:.1f} MB) "
            f"exceeds limit ({memory_bytes / 1024**2:.1f} MB)"
        )


def _parse_memory_string(memory_str: str) -> int:
    """
    Parse memory string like '1GB', '500MB' to bytes.
    
    Parameters
    ----------
    memory_str : str
        Memory string
        
    Returns
    -------
    int
        Memory in bytes
    """
    memory_str = memory_str.upper().strip()
    
    if memory_str.endswith('GB'):
        return int(float(memory_str[:-2]) * 1024**3)
    elif memory_str.endswith('MB'):
        return int(float(memory_str[:-2]) * 1024**2)
    elif memory_str.endswith('KB'):
        return int(float(memory_str[:-2]) * 1024)
    elif memory_str.endswith('B'):
        return int(memory_str[:-1])
    else:
        # Assume bytes if no suffix
        return int(memory_str)


def sanitize_column_names(columns: List[str]) -> List[str]:
    """
    Sanitize column names to prevent injection attacks.
    
    Parameters
    ----------
    columns : List[str]
        Column names to sanitize
        
    Returns
    -------
    List[str]
        Sanitized column names
    """
    import re
    
    sanitized = []
    for col in columns:
        # Remove potentially dangerous characters
        sanitized_col = re.sub(r'[^\w\-_\.]', '_', str(col))
        sanitized.append(sanitized_col)
    
    return sanitized


def validate_file_safety(filepath: str, allowed_extensions: Optional[List[str]] = None) -> None:
    """
    Validate file path for security.
    
    Parameters
    ----------
    filepath : str
        File path to validate
    allowed_extensions : List[str], optional
        Allowed file extensions
        
    Raises
    ------
    ValueError
        If file path is unsafe
    """
    import os
    from pathlib import Path
    
    path = Path(filepath)
    
    # Check for path traversal attempts
    if '..' in path.parts:
        raise ValueError("Path traversal detected in filepath")
    
    # Check file extension if specified
    if allowed_extensions:
        if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
            raise ValueError(f"File extension {path.suffix} not allowed. "
                           f"Allowed: {allowed_extensions}")
    
    # Check file size if it exists
    if path.exists():
        file_size = path.stat().st_size
        max_size = 500 * 1024 * 1024  # 500MB default limit
        if file_size > max_size:
            raise ValueError(f"File size ({file_size / 1024**2:.1f} MB) exceeds limit")


def detect_encoding(filepath: str, sample_size: int = 10000) -> str:
    """
    Detect file encoding safely.
    
    Parameters
    ----------
    filepath : str
        Path to file
    sample_size : int
        Number of bytes to sample for detection
        
    Returns
    -------
    str
        Detected encoding
    """
    import chardet
    
    validate_file_safety(filepath)
    
    with open(filepath, 'rb') as f:
        sample = f.read(sample_size)
        result = chardet.detect(sample)
        
    return result.get('encoding', 'utf-8')