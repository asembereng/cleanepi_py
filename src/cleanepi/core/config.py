"""
Configuration classes for data cleaning operations.

Uses Pydantic for validation and type safety.
"""

from datetime import date
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from pydantic import BaseModel, Field, validator


class DateConfig(BaseModel):
    """Configuration for date standardization operations."""

    target_columns: Optional[List[str]] = Field(
        None,
        description="List of column names to standardize. If None, auto-detect date columns",
    )
    formats: Optional[List[str]] = Field(
        None, description="Expected date formats. If None, use intelligent parsing"
    )
    timeframe: Optional[Tuple[str, str]] = Field(
        None,
        description="Valid date range as (start_date, end_date) strings in YYYY-MM-DD format",
    )
    error_tolerance: float = Field(
        0.4,
        ge=0.0,
        le=1.0,
        description="Proportion of unparseable dates to tolerate (0.0-1.0)",
    )
    orders: Optional[Dict[str, List[str]]] = Field(
        None, description="Custom date parsing orders by category"
    )

    @validator("timeframe")
    def validate_timeframe(cls, v):
        """Validate timeframe format and logic."""
        if v is not None:
            if len(v) != 2:
                raise ValueError("timeframe must be a tuple of two date strings")
            try:
                start_date = pd.to_datetime(v[0])
                end_date = pd.to_datetime(v[1])
                if start_date >= end_date:
                    raise ValueError("start_date must be before end_date")
            except Exception:
                raise ValueError("timeframe dates must be in YYYY-MM-DD format")
        return v


class SubjectIDConfig(BaseModel):
    """Configuration for subject ID validation and standardization."""

    target_columns: List[str] = Field(
        ..., description="Column names containing subject IDs"
    )
    prefix: Optional[str] = Field(None, description="Expected prefix for subject IDs")
    suffix: Optional[str] = Field(None, description="Expected suffix for subject IDs")
    range: Optional[Tuple[int, int]] = Field(
        None, description="Valid numeric range for ID numbers as (min, max)"
    )
    nchar: Optional[int] = Field(
        None, ge=1, description="Expected total character length"
    )
    pattern: Optional[str] = Field(
        None, description="Custom regex pattern for validation"
    )

    @validator("range")
    def validate_range(cls, v):
        """Validate ID range."""
        if v is not None:
            if len(v) != 2:
                raise ValueError("range must be a tuple of two integers")
            if v[0] >= v[1]:
                raise ValueError("range minimum must be less than maximum")
        return v


class MissingValueConfig(BaseModel):
    """Configuration for missing value replacement."""

    target_columns: Optional[List[str]] = Field(
        None, description="Columns to process. If None, process all columns"
    )
    na_strings: List[str] = Field(
        default_factory=lambda: ["-99", "N/A", "NULL", "", "missing", "unknown"],
        description="Strings to treat as missing values",
    )
    custom_na_by_column: Optional[Dict[str, List[str]]] = Field(
        None, description="Column-specific missing value strings"
    )


class DuplicateConfig(BaseModel):
    """Configuration for duplicate removal."""

    target_columns: Optional[List[str]] = Field(
        None, description="Columns to consider for duplicates. If None, use all columns"
    )
    subset: Optional[List[str]] = Field(
        None, description="Alias for target_columns for pandas compatibility"
    )
    keep: str = Field(
        "first",
        pattern="^(first|last|False)$",
        description="Which duplicates to keep: 'first', 'last', or False (remove all)",
    )


class ConstantConfig(BaseModel):
    """Configuration for constant column removal."""

    cutoff: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Proportion of values that must be constant to remove column",
    )
    exclude_columns: Optional[List[str]] = Field(
        None, description="Columns to exclude from constant checking"
    )


class NumericConfig(BaseModel):
    """Configuration for numeric conversion."""

    target_columns: List[str] = Field(..., description="Columns to convert to numeric")
    lang: str = Field(
        "en", description="Language for number word conversion (e.g., 'one' -> 1)"
    )
    errors: str = Field(
        "coerce",
        pattern="^(raise|coerce|ignore)$",
        description="How to handle conversion errors",
    )


class CleaningConfig(BaseModel):
    """Main configuration for all cleaning operations."""

    # Individual operation configs
    standardize_column_names: Union[bool, Dict[str, Any]] = Field(
        True, description="Whether to standardize column names or config dict"
    )
    replace_missing_values: Optional[MissingValueConfig] = Field(
        default_factory=MissingValueConfig,
        description="Missing value replacement configuration",
    )
    remove_duplicates: Optional[DuplicateConfig] = Field(
        default_factory=DuplicateConfig, description="Duplicate removal configuration"
    )
    remove_constants: Optional[ConstantConfig] = Field(
        default_factory=ConstantConfig,
        description="Constant column removal configuration",
    )
    standardize_dates: Optional[DateConfig] = Field(
        None, description="Date standardization configuration"
    )
    standardize_subject_ids: Optional[SubjectIDConfig] = Field(
        None, description="Subject ID validation configuration"
    )
    to_numeric: Optional[NumericConfig] = Field(
        None, description="Numeric conversion configuration"
    )
    dictionary: Optional[Dict[str, Dict[str, str]]] = Field(
        None, description="Dictionary for value replacement by column"
    )
    check_date_sequence: Optional[List[str]] = Field(
        None, description="Columns to check for proper date sequence"
    )

    # Global settings
    verbose: bool = Field(True, description="Whether to show progress messages")
    strict_validation: bool = Field(
        False, description="Whether to raise errors on validation failures"
    )
    max_memory_usage: Optional[str] = Field(
        None, description="Maximum memory usage (e.g., '1GB', '500MB')"
    )

    class Config:
        """Pydantic config."""

        extra = "forbid"
        validate_assignment = True


class WebConfig(BaseModel):
    """Configuration for web application settings."""

    max_file_size: int = Field(
        100 * 1024 * 1024, description="Maximum upload file size in bytes"  # 100MB
    )
    allowed_file_types: List[str] = Field(
        default_factory=lambda: [".csv", ".xlsx", ".parquet", ".json"],
        description="Allowed file extensions",
    )
    temp_dir: str = Field(
        "/tmp/cleanepi", description="Temporary directory for file processing"
    )
    enable_async: bool = Field(
        True, description="Enable async processing for large files"
    )
    chunk_size: int = Field(10000, description="Chunk size for processing large files")

    class Config:
        """Pydantic config."""

        extra = "forbid"
