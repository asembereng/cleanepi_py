"""Numeric conversion functionality."""

from typing import List
import pandas as pd
from loguru import logger

def convert_to_numeric(
    data: pd.DataFrame,
    target_columns: List[str],
    lang: str = "en",
    errors: str = "coerce"
) -> pd.DataFrame:
    """Convert columns to numeric."""
    logger.info("Numeric conversion not yet implemented")
    return data