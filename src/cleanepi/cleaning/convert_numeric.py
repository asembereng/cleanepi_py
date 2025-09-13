"""Numeric conversion functionality."""

from typing import List, Dict, Optional, Union, Any
import pandas as pd
import numpy as np
import re
from loguru import logger

from ..utils.validation import validate_dataframe, validate_columns_exist


# Number word mappings for different languages
NUMBER_WORDS = {
    'en': {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
        'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
        'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000, 'million': 1000000
    },
    'es': {
        'cero': 0, 'uno': 1, 'dos': 2, 'tres': 3, 'cuatro': 4, 'cinco': 5,
        'seis': 6, 'siete': 7, 'ocho': 8, 'nueve': 9, 'diez': 10,
        'once': 11, 'doce': 12, 'trece': 13, 'catorce': 14, 'quince': 15,
        'dieciséis': 16, 'diecisiete': 17, 'dieciocho': 18, 'diecinueve': 19,
        'veinte': 20, 'treinta': 30, 'cuarenta': 40, 'cincuenta': 50,
        'sesenta': 60, 'setenta': 70, 'ochenta': 80, 'noventa': 90,
        'cien': 100, 'mil': 1000, 'millón': 1000000
    },
    'fr': {
        'zéro': 0, 'un': 1, 'deux': 2, 'trois': 3, 'quatre': 4, 'cinq': 5,
        'six': 6, 'sept': 7, 'huit': 8, 'neuf': 9, 'dix': 10,
        'onze': 11, 'douze': 12, 'treize': 13, 'quatorze': 14, 'quinze': 15,
        'seize': 16, 'dix-sept': 17, 'dix-huit': 18, 'dix-neuf': 19,
        'vingt': 20, 'trente': 30, 'quarante': 40, 'cinquante': 50,
        'soixante': 60, 'soixante-dix': 70, 'quatre-vingts': 80, 'quatre-vingt-dix': 90,
        'cent': 100, 'mille': 1000, 'million': 1000000
    }
}


def convert_to_numeric(
    data: pd.DataFrame,
    target_columns: List[str],
    lang: str = "en",
    errors: str = "coerce"
) -> pd.DataFrame:
    """
    Convert columns to numeric with intelligent parsing.
    
    This function converts text and mixed data to numeric by:
    - Handling number words in multiple languages
    - Cleaning numeric strings (removing currency, commas, etc.)
    - Converting percentages
    - Handling scientific notation
    - Dealing with range values (e.g., "10-15")
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
    target_columns : List[str]
        Columns to convert to numeric
    lang : str, default "en"
        Language for number word conversion ("en", "es", "fr")
    errors : str, default "coerce"
        How to handle conversion errors:
        - "raise": raise exception on errors
        - "coerce": convert errors to NaN
        - "ignore": return original data on errors
        
    Returns
    -------
    pd.DataFrame
        DataFrame with converted numeric columns
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'age': ['25', 'thirty', '45.5', 'unknown'],
    ...     'income': ['$50,000', '75000', 'sixty thousand', 'N/A'],
    ...     'percentage': ['85%', '90.5%', 'fifty percent', 'missing']
    ... })
    >>> result = convert_to_numeric(df, ['age', 'income', 'percentage'])
    >>> result.dtypes
    age           float64
    income        float64  
    percentage    float64
    dtype: object
    """
    validate_dataframe(data)
    validate_columns_exist(data, target_columns, "convert_to_numeric")
    
    # Make a copy to avoid modifying original
    result = data.copy()
    
    # Validate language
    if lang not in NUMBER_WORDS:
        logger.warning(f"Language '{lang}' not supported, using 'en'")
        lang = 'en'
    
    logger.info(f"Converting {len(target_columns)} columns to numeric (language: {lang})")
    
    conversion_summary = {}
    
    for col in target_columns:
        logger.info(f"Converting column: {col}")
        
        # Skip if already numeric
        if pd.api.types.is_numeric_dtype(result[col]):
            logger.info(f"Column {col} is already numeric, skipping")
            conversion_summary[col] = {
                'original_type': str(result[col].dtype),
                'converted': False,
                'reason': 'already_numeric'
            }
            continue
        
        original_series = result[col].copy()
        total_values = len(original_series.dropna())
        
        if total_values == 0:
            logger.warning(f"Column {col} has no non-null values")
            conversion_summary[col] = {
                'original_type': str(original_series.dtype),
                'converted': False,
                'reason': 'no_data'
            }
            continue
        
        try:
            # Convert the column
            converted_series = _convert_series_to_numeric(original_series, lang)
            
            # Handle errors based on errors parameter
            if errors == "raise":
                # Check if any values failed to convert (excluding originally null values)
                non_null_original = original_series.notna()
                failed_conversions = non_null_original & converted_series.isna()
                if failed_conversions.any():
                    failed_values = original_series[failed_conversions].tolist()
                    raise ValueError(f"Failed to convert values in column {col}: {failed_values}")
            
            elif errors == "ignore":
                # Check conversion success rate
                non_null_original = original_series.notna()
                successful_conversions = non_null_original & converted_series.notna()
                success_rate = successful_conversions.sum() / non_null_original.sum()
                
                if success_rate < 0.5:  # Less than 50% success
                    logger.warning(f"Low conversion success rate for {col} ({success_rate:.1%}), keeping original")
                    conversion_summary[col] = {
                        'original_type': str(original_series.dtype),
                        'converted': False,
                        'reason': 'low_success_rate',
                        'success_rate': success_rate
                    }
                    continue
            
            # Update result
            result[col] = converted_series
            
            # Calculate success metrics
            successful_count = converted_series.notna().sum()
            success_rate = successful_count / total_values if total_values > 0 else 0
            
            conversion_summary[col] = {
                'original_type': str(original_series.dtype),
                'new_type': str(converted_series.dtype),
                'converted': True,
                'total_values': total_values,
                'successful_conversions': successful_count,
                'success_rate': success_rate
            }
            
            logger.info(
                f"Column {col}: {successful_count}/{total_values} values converted "
                f"({success_rate:.1%} success rate)"
            )
            
        except Exception as e:
            if errors == "raise":
                raise
            elif errors == "ignore":
                logger.warning(f"Failed to convert column {col}: {e}")
                conversion_summary[col] = {
                    'original_type': str(original_series.dtype),
                    'converted': False,
                    'reason': 'conversion_error',
                    'error': str(e)
                }
            else:  # errors == "coerce"
                logger.warning(f"Error converting column {col}: {e}, setting to NaN")
                result[col] = pd.Series(np.nan, index=result.index)
                conversion_summary[col] = {
                    'original_type': str(original_series.dtype),
                    'new_type': 'float64',
                    'converted': True,
                    'total_values': total_values,
                    'successful_conversions': 0,
                    'success_rate': 0.0,
                    'error': str(e)
                }
    
    # Log overall summary
    total_columns = len(target_columns)
    converted_columns = sum(1 for s in conversion_summary.values() if s['converted'])
    logger.info(f"Successfully converted {converted_columns}/{total_columns} columns to numeric")
    
    return result


def _convert_series_to_numeric(series: pd.Series, lang: str) -> pd.Series:
    """
    Convert a pandas Series to numeric using intelligent parsing.
    
    Parameters
    ----------
    series : pd.Series
        Series to convert
    lang : str
        Language for number word conversion
        
    Returns
    -------
    pd.Series
        Converted numeric series
    """
    result = pd.Series(index=series.index, dtype='float64')
    
    for idx, value in series.items():
        if pd.isna(value):
            result.iloc[idx] = np.nan
        else:
            converted_value = _convert_value_to_numeric(value, lang)
            result.iloc[idx] = converted_value
    
    return result


def _convert_value_to_numeric(value: Any, lang: str) -> Union[float, None]:
    """
    Convert a single value to numeric.
    
    Parameters
    ----------
    value : Any
        Value to convert
    lang : str
        Language for number word conversion
        
    Returns
    -------
    float or None
        Converted numeric value or None if conversion fails
    """
    if pd.isna(value):
        return np.nan
    
    # If already numeric, return as float
    if isinstance(value, (int, float)):
        return float(value)
    
    # Convert to string and clean
    str_value = str(value).strip()
    
    if not str_value:
        return np.nan
    
    # Try direct pandas conversion first
    try:
        return pd.to_numeric(str_value)
    except:
        pass
    
    # Clean the string and try various parsing methods
    cleaned_value = _clean_numeric_string(str_value)
    
    # Try pandas conversion on cleaned string
    try:
        return pd.to_numeric(cleaned_value)
    except:
        pass
    
    # Try word-to-number conversion
    try:
        word_number = _words_to_number(str_value, lang)
        if word_number is not None:
            return float(word_number)
    except:
        pass
    
    # Try percentage conversion
    try:
        percentage = _parse_percentage(str_value)
        if percentage is not None:
            return percentage
    except:
        pass
    
    # Try range conversion (take midpoint)
    try:
        range_midpoint = _parse_range(str_value)
        if range_midpoint is not None:
            return range_midpoint
    except:
        pass
    
    # If all methods fail, return NaN
    return np.nan


def _clean_numeric_string(value: str) -> str:
    """
    Clean a string to extract numeric parts.
    
    Parameters
    ----------
    value : str
        String to clean
        
    Returns
    -------
    str
        Cleaned string
    """
    # Remove common currency symbols and units
    cleaned = value
    
    # Remove currency symbols
    currency_symbols = ['$', '€', '£', '¥', '₹', '¢']
    for symbol in currency_symbols:
        cleaned = cleaned.replace(symbol, '')
    
    # Remove common units
    units = ['kg', 'lbs', 'cm', 'ft', 'in', 'm', 'km', 'mi']
    for unit in units:
        cleaned = re.sub(rf'\b{unit}\b', '', cleaned, flags=re.IGNORECASE)
    
    # Remove percentage symbols (but keep the number)
    cleaned = cleaned.replace('%', '')
    
    # Remove thousand separators (commas)
    cleaned = cleaned.replace(',', '')
    
    # Remove extra whitespace
    cleaned = cleaned.strip()
    
    return cleaned


def _words_to_number(text: str, lang: str) -> Optional[int]:
    """
    Convert number words to numeric values.
    
    Parameters
    ----------
    text : str
        Text containing number words
    lang : str
        Language code
        
    Returns
    -------
    int or None
        Converted number or None if conversion fails
    """
    if lang not in NUMBER_WORDS:
        return None
    
    word_map = NUMBER_WORDS[lang]
    text_lower = text.lower().strip()
    
    # Direct word lookup
    if text_lower in word_map:
        return word_map[text_lower]
    
    # Handle compound words (basic implementation)
    # This is a simplified version - full implementation would need more complex parsing
    words = re.findall(r'\b\w+\b', text_lower)
    total = 0
    current = 0
    
    for word in words:
        if word in word_map:
            value = word_map[word]
            if value == 100:
                current *= 100
            elif value == 1000 or value == 1000000:
                current *= value
                total += current
                current = 0
            else:
                current += value
    
    total += current
    
    return total if total > 0 else None


def _parse_percentage(text: str) -> Optional[float]:
    """
    Parse percentage values.
    
    Parameters
    ----------
    text : str
        Text that might contain a percentage
        
    Returns
    -------
    float or None
        Percentage as decimal (e.g., 85% -> 0.85) or None
    """
    text_lower = text.lower().strip()
    
    # Check for percentage symbol
    if '%' in text_lower or 'percent' in text_lower:
        # Extract numeric part
        numeric_match = re.search(r'(\d+(?:\.\d+)?)', text_lower)
        if numeric_match:
            try:
                percentage_value = float(numeric_match.group(1))
                return percentage_value / 100.0  # Convert to decimal
            except ValueError:
                pass
    
    return None


def _parse_range(text: str) -> Optional[float]:
    """
    Parse range values and return the midpoint.
    
    Parameters
    ----------
    text : str
        Text that might contain a range (e.g., "10-15", "5 to 10")
        
    Returns
    -------
    float or None
        Midpoint of the range or None
    """
    # Pattern for ranges like "10-15", "5 to 10", "between 20 and 30"
    range_patterns = [
        r'(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)',  # 10-15, 10–15
        r'(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)',     # 5 to 10
        r'between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)',  # between 20 and 30
    ]
    
    text_lower = text.lower().strip()
    
    for pattern in range_patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                start = float(match.group(1))
                end = float(match.group(2))
                return (start + end) / 2.0
            except ValueError:
                continue
    
    return None