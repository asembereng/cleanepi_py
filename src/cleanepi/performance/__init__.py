"""
Performance and scalability features for cleanepi.

This module provides high-performance and large-scale data processing capabilities
including Dask integration, async processing, streaming support, and distributed
processing.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dask_processing import DaskCleaner
    from .async_processing import AsyncCleaner
    from .streaming import StreamingCleaner
    from .distributed import DistributedCleaner

__all__ = [
    "DaskCleaner",
    "AsyncCleaner", 
    "StreamingCleaner",
    "DistributedCleaner",
]


def get_dask_cleaner():
    """Get Dask-based cleaner (requires dask[dataframe] to be installed)."""
    try:
        from .dask_processing import DaskCleaner
        return DaskCleaner
    except ImportError as e:
        raise ImportError(
            "Dask integration requires 'dask[dataframe]' to be installed. "
            "Install with: pip install 'cleanepi-python[performance]'"
        ) from e


def get_async_cleaner():
    """Get async-based cleaner (requires aiofiles to be installed)."""
    try:
        from .async_processing import AsyncCleaner
        return AsyncCleaner
    except ImportError as e:
        raise ImportError(
            "Async processing requires 'aiofiles' to be installed. "
            "Install with: pip install 'cleanepi-python[async]'"
        ) from e


def get_streaming_cleaner():
    """Get streaming data cleaner."""
    from .streaming import StreamingCleaner
    return StreamingCleaner


def get_distributed_cleaner():
    """Get distributed processing cleaner (requires dask.distributed)."""
    try:
        from .distributed import DistributedCleaner
        return DistributedCleaner
    except ImportError as e:
        raise ImportError(
            "Distributed processing requires 'dask.distributed' to be installed. "
            "Install with: pip install 'cleanepi-python[performance]'"
        ) from e