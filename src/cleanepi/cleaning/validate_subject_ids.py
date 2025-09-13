"""Subject ID validation functionality."""

from typing import List, Optional, Tuple

import pandas as pd
from loguru import logger


def check_subject_ids(
    data: pd.DataFrame,
    target_columns: List[str],
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    range: Optional[Tuple[int, int]] = None,
    nchar: Optional[int] = None,
    pattern: Optional[str] = None,
) -> pd.DataFrame:
    """Check and validate subject IDs."""
    logger.info("Subject ID validation not yet implemented")
    return data
