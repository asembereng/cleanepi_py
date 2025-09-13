"""
Validation utilities for input data and configuration.
"""

from typing import Any, List, Optional

import pandas as pd
from loguru import logger

from ..core.config import CleaningConfig


def validate_dataframe(data: Any, min_rows: int = 1, min_cols: int = 1) -> None:
    """
    Validate that input is a proper pandas DataFrame.

    Parameters
    ----------
    data : Any
        Input to validate
    min_rows : int
        Minimum required rows
    min_cols : int
        Minimum required columns

    Raises
    ------
    TypeError
        If data is not a pandas DataFrame
    ValueError
        If DataFrame doesn't meet minimum requirements
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(data)}")

    if data.empty:
        raise ValueError("DataFrame is empty")

    if len(data) < min_rows:
        raise ValueError(f"DataFrame has {len(data)} rows, minimum {min_rows} required")

    if len(data.columns) < min_cols:
        raise ValueError(
            f"DataFrame has {len(data.columns)} columns, minimum {min_cols} required"
        )

    logger.debug(f"DataFrame validation passed: shape {data.shape}")


def validate_config(config: CleaningConfig) -> None:
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
    if (
        config.standardize_subject_ids
        and not config.standardize_subject_ids.target_columns
    ):
        raise ValueError("standardize_subject_ids requires target_columns")

    if config.to_numeric and not config.to_numeric.target_columns:
        raise ValueError("to_numeric requires target_columns")

    logger.debug("Configuration validation passed")


def validate_columns_exist(
    data: pd.DataFrame, columns: List[str], operation: str = ""
) -> None:
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
    operation: str = "",
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
        if not any(
            pd.api.types.is_dtype_equal(col_type, expected_type)
            for expected_type in expected_types
        ):
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

    if memory_str.endswith("GB"):
        return int(float(memory_str[:-2]) * 1024**3)
    elif memory_str.endswith("MB"):
        return int(float(memory_str[:-2]) * 1024**2)
    elif memory_str.endswith("KB"):
        return int(float(memory_str[:-2]) * 1024)
    elif memory_str.endswith("B"):
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
        sanitized_col = re.sub(r"[^\w\-_\.]", "_", str(col))
        sanitized.append(sanitized_col)

    return sanitized


def validate_file_safety(
    filepath: str, allowed_extensions: Optional[List[str]] = None
) -> None:
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
    if ".." in path.parts:
        raise ValueError("Path traversal detected in filepath")

    # Check file extension if specified
    if allowed_extensions:
        if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
            raise ValueError(
                f"File extension {path.suffix} not allowed. "
                f"Allowed: {allowed_extensions}"
            )

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

    with open(filepath, "rb") as f:
        sample = f.read(sample_size)
        result = chardet.detect(sample)

    return result.get("encoding", "utf-8")
