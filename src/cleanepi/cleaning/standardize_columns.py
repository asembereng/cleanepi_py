"""
Column name standardization functionality.

Cleans and standardizes column names to follow consistent naming conventions.
"""

from typing import Dict, List, Optional, Union
import pandas as pd
import re
from loguru import logger

from ..utils.validation import validate_dataframe


def standardize_column_names(
    data: pd.DataFrame,
    keep: Optional[List[str]] = None,
    rename: Optional[Dict[str, str]] = None,
    style: str = "snake_case",
    remove_special_chars: bool = True,
    max_length: Optional[int] = None
) -> pd.DataFrame:
    """
    Standardize column names to follow consistent naming conventions.
    
    This function cleans column names by:
    - Converting to consistent case (snake_case, camelCase, etc.)
    - Removing or replacing special characters
    - Handling duplicates
    - Applying custom renaming rules
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
    keep : List[str], optional
        Column names to keep unchanged
    rename : Dict[str, str], optional  
        Custom column renaming mapping {old_name: new_name}
    style : str, default "snake_case"
        Naming style: "snake_case", "camelCase", "PascalCase", "kebab-case"
    remove_special_chars : bool, default True
        Whether to remove special characters
    max_length : int, optional
        Maximum length for column names
        
    Returns
    -------
    pd.DataFrame
        DataFrame with standardized column names
        
    Examples
    --------
    >>> df = pd.DataFrame({'Date of Birth': [1, 2], 'Patient ID': [3, 4]})
    >>> clean_df = standardize_column_names(df)
    >>> list(clean_df.columns)
    ['date_of_birth', 'patient_id']
    
    >>> # Custom renaming
    >>> clean_df = standardize_column_names(df, rename={'Date of Birth': 'dob'})
    >>> list(clean_df.columns)  
    ['dob', 'patient_id']
    """
    validate_dataframe(data)
    
    # Make a copy to avoid modifying original
    result = data.copy()
    
    # Get original column names
    original_columns = list(data.columns)
    keep = keep or []
    rename = rename or {}
    
    logger.info(f"Standardizing {len(original_columns)} column names")
    
    # Apply standardization
    new_columns = []
    for col in original_columns:
        if col in keep:
            # Keep unchanged
            new_col = col
        elif col in rename:
            # Apply custom rename
            new_col = rename[col]
        else:
            # Apply standardization
            new_col = _standardize_single_name(
                col, style, remove_special_chars, max_length
            )
        
        new_columns.append(new_col)
    
    # Handle duplicates
    new_columns = _handle_duplicate_names(new_columns)
    
    # Apply new column names
    result.columns = new_columns
    
    # Log changes
    changes = [(old, new) for old, new in zip(original_columns, new_columns) 
               if old != new]
    if changes:
        logger.info(f"Renamed {len(changes)} columns")
        for old, new in changes[:5]:  # Show first 5 changes
            logger.debug(f"  '{old}' -> '{new}'")
        if len(changes) > 5:
            logger.debug(f"  ... and {len(changes) - 5} more")
    else:
        logger.info("No column names changed")
    
    return result


def _standardize_single_name(
    name: str,
    style: str,
    remove_special_chars: bool,
    max_length: Optional[int]
) -> str:
    """
    Standardize a single column name.
    
    Parameters
    ----------
    name : str
        Original column name
    style : str
        Target naming style
    remove_special_chars : bool
        Whether to remove special characters
    max_length : int, optional
        Maximum length
        
    Returns
    -------
    str
        Standardized column name
    """
    # Convert to string and strip whitespace
    clean_name = str(name).strip()
    
    # Remove or replace special characters
    if remove_special_chars:
        # Keep only alphanumeric and common separators
        clean_name = re.sub(r'[^\w\s\-_.]', '', clean_name)
    
    # Replace multiple spaces/separators with single space
    clean_name = re.sub(r'[\s\-_.]+', ' ', clean_name)
    
    # Apply naming style
    if style == "snake_case":
        clean_name = _to_snake_case(clean_name)
    elif style == "camelCase":
        clean_name = _to_camel_case(clean_name)
    elif style == "PascalCase":
        clean_name = _to_pascal_case(clean_name)
    elif style == "kebab-case":
        clean_name = _to_kebab_case(clean_name)
    else:
        logger.warning(f"Unknown style '{style}', using snake_case")
        clean_name = _to_snake_case(clean_name)
    
    # Apply length limit
    if max_length and len(clean_name) > max_length:
        clean_name = clean_name[:max_length]
        # Ensure it doesn't end with separator
        clean_name = clean_name.rstrip('_-.')
    
    # Ensure name is not empty
    if not clean_name:
        clean_name = "column"
    
    # Ensure name doesn't start with number
    if clean_name[0].isdigit():
        clean_name = "col_" + clean_name
    
    return clean_name


def _to_snake_case(text: str) -> str:
    """Convert text to snake_case."""
    # Handle camelCase
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', text)
    # Convert to lowercase and replace spaces/separators with underscores
    text = re.sub(r'[\s\-\.]+', '_', text.lower())
    # Remove multiple underscores
    text = re.sub(r'_+', '_', text)
    # Remove leading/trailing underscores
    return text.strip('_')


def _to_camel_case(text: str) -> str:
    """Convert text to camelCase."""
    words = re.split(r'[\s\-_.]+', text.lower())
    if not words:
        return text
    
    # First word lowercase, rest title case
    return words[0] + ''.join(word.capitalize() for word in words[1:])


def _to_pascal_case(text: str) -> str:
    """Convert text to PascalCase."""
    words = re.split(r'[\s\-_.]+', text.lower())
    return ''.join(word.capitalize() for word in words)


def _to_kebab_case(text: str) -> str:
    """Convert text to kebab-case."""
    # Handle camelCase
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1-\2', text)
    # Convert to lowercase and replace spaces/separators with hyphens
    text = re.sub(r'[\s_\.]+', '-', text.lower())
    # Remove multiple hyphens
    text = re.sub(r'-+', '-', text)
    # Remove leading/trailing hyphens
    return text.strip('-')


def _handle_duplicate_names(names: List[str]) -> List[str]:
    """
    Handle duplicate column names by adding suffixes.
    
    Parameters
    ----------
    names : List[str]
        List of column names
        
    Returns
    -------
    List[str]
        List with unique column names
    """
    seen = {}
    result = []
    
    for name in names:
        if name not in seen:
            seen[name] = 0
            result.append(name)
        else:
            seen[name] += 1
            result.append(f"{name}_{seen[name]}")
    
    return result


def suggest_column_renames(data: pd.DataFrame) -> Dict[str, str]:
    """
    Suggest column renames based on common patterns.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
        
    Returns
    -------
    Dict[str, str]
        Suggested renaming mapping
    """
    suggestions = {}
    common_mappings = {
        # Date patterns
        r'date[\s_\-]*of[\s_\-]*birth': 'date_of_birth',
        r'dob': 'date_of_birth',
        r'birth[\s_\-]*date': 'date_of_birth',
        r'admission[\s_\-]*date': 'admission_date',
        r'discharge[\s_\-]*date': 'discharge_date',
        
        # ID patterns  
        r'patient[\s_\-]*id': 'patient_id',
        r'subject[\s_\-]*id': 'subject_id',
        r'study[\s_\-]*id': 'study_id',
        
        # Common fields
        r'first[\s_\-]*name': 'first_name',
        r'last[\s_\-]*name': 'last_name',
        r'family[\s_\-]*name': 'last_name',
        r'given[\s_\-]*name': 'first_name',
        
        # Demographics
        r'sex|gender': 'sex',
        r'age': 'age',
        r'weight': 'weight',
        r'height': 'height',
    }
    
    for col in data.columns:
        col_lower = str(col).lower()
        for pattern, suggestion in common_mappings.items():
            if re.search(pattern, col_lower):
                suggestions[col] = suggestion
                break
    
    return suggestions


def clean_column_names_interactive(data: pd.DataFrame) -> pd.DataFrame:
    """
    Interactive column name cleaning with user prompts.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
        
    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned column names
    """
    print("Column Name Cleaning")
    print("=" * 40)
    
    # Show current columns
    print(f"\nCurrent columns ({len(data.columns)}):")
    for i, col in enumerate(data.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Get suggestions
    suggestions = suggest_column_renames(data)
    if suggestions:
        print(f"\nSuggested renames:")
        for old, new in suggestions.items():
            print(f"  '{old}' -> '{new}'")
        
        use_suggestions = input("\nUse suggestions? (y/n): ").lower().startswith('y')
        if use_suggestions:
            return standardize_column_names(data, rename=suggestions)
    
    # Manual mode
    print("\nEnter custom renames (format: old_name=new_name, or press Enter to skip):")
    custom_renames = {}
    
    for col in data.columns:
        new_name = input(f"  '{col}' -> ").strip()
        if new_name:
            custom_renames[col] = new_name
    
    return standardize_column_names(data, rename=custom_renames)