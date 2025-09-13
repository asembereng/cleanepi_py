"""Convert columns into numeric format."""

import pandas as pd
import numpy as np
import re
from typing import List, Optional, Union
from .utils import (
    get_target_column_names,
    add_to_report,
    retrieve_column_names,
    validate_dataframe_input
)


def convert_to_numeric(data: pd.DataFrame,
                      target_columns: Optional[List[str]] = None,
                      lang: str = "en") -> pd.DataFrame:
    """
    Convert columns into numeric format.

    When this function is invoked without specifying the column names to be
    converted, it identifies columns where the proportion of numeric values
    is at least twice the percentage of character values and performs the
    conversion in them.

    Args:
        data: The input DataFrame
        target_columns: A list of the target column names to convert.
        lang: A string with the text's language. Currently one of "en", "fr", "es".

    Returns:
        A DataFrame wherein all the specified or detected columns have been
        transformed into numeric format after the conversion process.
        
    Examples:
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        ...     'age': ['25', 'thirty', '35', 'forty'],
        ...     'score': ['10', '20', 'thirty', '40']
        ... })
        >>> cleaned_data = convert_to_numeric(data, target_columns=['age'], lang='en')
        >>> print(cleaned_data)
    """
    # Validate input
    data = validate_dataframe_input(data)
    
    # Validate language
    if lang not in ["en", "fr", "es"]:
        raise ValueError("Language must be one of 'en', 'fr', 'es'")
    
    # Make a copy to avoid modifying the original data
    data_copy = data.copy()
    
    # If no target columns specified, auto-detect
    if target_columns is None:
        scan_result = scan_data(data_copy)
        target_columns = detect_to_numeric_columns(scan_result, data_copy)
    
    # Get the correct column names
    target_columns = retrieve_column_names(data_copy, target_columns)
    cols = get_target_column_names(data_copy, target_columns, cols=None)
    
    if len(cols) == 0:
        print("Found one or more columns with insufficient numeric values for automatic conversion.")
        print("The percentage of character values must be less than twice the numeric values "
              "for a column to be considered for automatic conversion.")
        print("Please specify names of the columns to convert into numeric using target_columns.")
        return data_copy
    
    # Convert each target column
    converted_columns = []
    for col in cols:
        if col in data_copy.columns:
            original_col = data_copy[col].copy()
            converted_col = numberize_column(data_copy[col], lang=lang)
            data_copy[col] = converted_col
            converted_columns.append(col)
    
    # Add report information
    if converted_columns:
        data_copy = add_to_report(
            data_copy,
            key="converted_into_numeric",
            value=converted_columns
        )
        print(f"Converted the following columns into numeric: {converted_columns}")
    
    return data_copy


def scan_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Scan data to identify column types and characteristics.
    
    Args:
        data: Input DataFrame to scan
        
    Returns:
        DataFrame with scanning results
    """
    results = []
    
    for col in data.columns:
        series = data[col]
        
        # Count different types
        total_count = len(series)
        missing_count = series.isnull().sum()
        
        # For non-null values, categorize them
        non_null = series.dropna()
        numeric_count = 0
        character_count = 0
        
        for value in non_null:
            if pd.api.types.is_numeric_dtype(type(value)):
                numeric_count += 1
            elif isinstance(value, str):
                # Try to convert to numeric
                try:
                    float(value)
                    numeric_count += 1
                except (ValueError, TypeError):
                    character_count += 1
            else:
                character_count += 1
        
        results.append({
            'Field_names': col,
            'missing': missing_count,
            'numeric': numeric_count,
            'character': character_count
        })
    
    return pd.DataFrame(results)


def detect_to_numeric_columns(scan_result: pd.DataFrame, data: pd.DataFrame) -> List[str]:
    """
    Detect the numeric columns that appear as characters due to the presence of
    some character values in the column.
    
    Args:
        scan_result: DataFrame that corresponds to the result from scan_data function
        data: Original data for reference
        
    Returns:
        List of column names to be converted into numeric
    """
    to_numeric = []
    
    for _, row in scan_result.iterrows():
        col_name = row['Field_names']
        numeric_count = row['numeric']
        character_count = row['character']
        
        # Check if both numeric and character values exist
        if numeric_count > 0 and character_count > 0:
            # Character count should be <= 2 * numeric count for conversion
            if character_count <= (2 * numeric_count):
                to_numeric.append(col_name)
            else:
                print(f"Found {numeric_count} numeric values in {col_name}.")
                print("Please consider the following options:")
                print("* Converting characters into numeric")
                print("* Replacing the numeric values by NA using the replace_missing_values function.")
    
    if to_numeric:
        print(f"The following columns will be converted into numeric: {to_numeric}")
    
    return to_numeric


def numberize_column(series: pd.Series, lang: str = "en") -> pd.Series:
    """
    Convert text numbers to numeric values in a pandas Series.
    
    Args:
        series: Input pandas Series
        lang: Language for number word recognition ("en", "fr", "es")
        
    Returns:
        Series with text numbers converted to numeric
    """
    result = series.copy()
    
    # Number word mappings for different languages
    number_words = {
        "en": {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
            "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
            "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
            "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000, "million": 1000000
        },
        "fr": {
            "zéro": 0, "un": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5,
            "six": 6, "sept": 7, "huit": 8, "neuf": 9, "dix": 10,
            "onze": 11, "douze": 12, "treize": 13, "quatorze": 14, "quinze": 15,
            "seize": 16, "dix-sept": 17, "dix-huit": 18, "dix-neuf": 19, "vingt": 20,
            "trente": 30, "quarante": 40, "cinquante": 50, "soixante": 60, "soixante-dix": 70,
            "quatre-vingts": 80, "quatre-vingt-dix": 90, "cent": 100, "mille": 1000, "million": 1000000
        },
        "es": {
            "cero": 0, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
            "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
            "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
            "dieciséis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20,
            "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70,
            "ochenta": 80, "noventa": 90, "cien": 100, "mil": 1000, "millón": 1000000
        }
    }
    
    word_map = number_words.get(lang, number_words["en"])
    
    for idx, value in series.items():
        if pd.isna(value):
            continue
            
        # Convert to string and clean
        str_value = str(value).strip().lower()
        
        # Try direct numeric conversion first
        try:
            result.iloc[idx] = pd.to_numeric(str_value)
            continue
        except (ValueError, TypeError):
            pass
        
        # Try word-to-number conversion
        if str_value in word_map:
            result.iloc[idx] = word_map[str_value]
            continue
        
        # For more complex number words, try to parse them
        # This is a simplified version - a full implementation would be more complex
        numeric_value = parse_number_words(str_value, word_map)
        if numeric_value is not None:
            result.iloc[idx] = numeric_value
    
    return result


def parse_number_words(text: str, word_map: dict) -> Optional[float]:
    """
    Parse number words into numeric values.
    
    Args:
        text: Text containing number words
        word_map: Dictionary mapping words to numbers
        
    Returns:
        Numeric value if parsing successful, None otherwise
    """
    # This is a simplified implementation
    # A full implementation would handle complex expressions like "twenty-five"
    
    # Remove common punctuation and split
    cleaned_text = re.sub(r'[^\w\s-]', ' ', text)
    words = cleaned_text.split()
    
    total = 0
    current = 0
    
    for word in words:
        word = word.strip('-').lower()
        if word in word_map:
            value = word_map[word]
            if value == 100:
                current *= 100
            elif value >= 1000:
                total += current * value
                current = 0
            else:
                current += value
        else:
            # If we encounter a word we don't recognize, return None
            return None
    
    return total + current if total + current > 0 else None