"""
Performance monitoring and optimization utilities.

Implements standards from terms_of_reference.md:
- Memory efficiency for datasets up to 10x available RAM through chunking
- Linear performance scaling with data size for core operations
- Configurable memory and CPU limits for different deployment scenarios
"""

import time
import gc
import warnings
from typing import Dict, Any, Optional, Iterator, Callable, Union
from functools import wraps
from contextlib import contextmanager
import pandas as pd
import numpy as np
from loguru import logger

from .validation import get_memory_usage


class PerformanceMonitor:
    """Monitor and track performance metrics for data cleaning operations."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        self.start_memory = None
    
    def start_operation(self, operation_name: str) -> None:
        """Start monitoring an operation."""
        self.metrics[operation_name] = {
            'start_time': time.time(),
            'start_memory': get_memory_usage(),
            'end_time': None,
            'end_memory': None,
            'duration': None,
            'memory_delta': None,
            'peak_memory': None
        }
        logger.debug(f"Started monitoring: {operation_name}")
    
    def end_operation(self, operation_name: str) -> Dict[str, Any]:
        """End monitoring an operation and return metrics."""
        if operation_name not in self.metrics:
            logger.warning(f"Operation {operation_name} was not started")
            return {}
        
        end_time = time.time()
        end_memory = get_memory_usage()
        
        metrics = self.metrics[operation_name]
        metrics['end_time'] = end_time
        metrics['end_memory'] = end_memory
        metrics['duration'] = end_time - metrics['start_time']
        metrics['memory_delta'] = end_memory['rss_mb'] - metrics['start_memory']['rss_mb']
        
        logger.info(
            f"Operation {operation_name}: "
            f"{metrics['duration']:.2f}s, "
            f"memory delta: {metrics['memory_delta']:+.1f}MB"
        )
        
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all monitored operations."""
        total_duration = sum(
            m.get('duration', 0) for m in self.metrics.values() 
            if m.get('duration') is not None
        )
        
        return {
            'operations': len(self.metrics),
            'total_duration': total_duration,
            'metrics': self.metrics
        }


def performance_monitor(operation_name: Optional[str] = None):
    """
    Decorator to monitor performance of functions.
    
    Parameters
    ----------
    operation_name : str, optional
        Name of the operation for logging
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            
            start_time = time.time()
            start_memory = get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                
                end_time = time.time()
                end_memory = get_memory_usage()
                
                duration = end_time - start_time
                memory_delta = end_memory['rss_mb'] - start_memory['rss_mb']
                
                logger.info(
                    f"Performance: {name} completed in {duration:.2f}s "
                    f"(memory delta: {memory_delta:+.1f}MB)"
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Performance: {name} failed after {time.time() - start_time:.2f}s")
                raise
        
        return wrapper
    return decorator


@contextmanager
def memory_limit_context(max_memory_mb: float):
    """
    Context manager to enforce memory limits during operations.
    
    Parameters
    ----------
    max_memory_mb : float
        Maximum allowed memory usage in megabytes
        
    Raises
    ------
    MemoryError
        If memory usage exceeds limit
    """
    initial_memory = get_memory_usage()
    
    try:
        yield
        
    finally:
        current_memory = get_memory_usage()
        if current_memory['rss_mb'] > max_memory_mb:
            # Try to free memory
            gc.collect()
            current_memory = get_memory_usage()
            
            if current_memory['rss_mb'] > max_memory_mb:
                raise MemoryError(
                    f"Memory usage ({current_memory['rss_mb']:.1f} MB) "
                    f"exceeds limit ({max_memory_mb} MB)"
                )


def chunk_dataframe(
    df: pd.DataFrame, 
    chunk_size: Optional[int] = None,
    max_memory_mb: Optional[float] = None
) -> Iterator[pd.DataFrame]:
    """
    Chunk DataFrame for memory-efficient processing.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to chunk
    chunk_size : int, optional
        Number of rows per chunk
    max_memory_mb : float, optional
        Maximum memory per chunk in megabytes
        
    Yields
    ------
    pd.DataFrame
        DataFrame chunks
    """
    if chunk_size is None and max_memory_mb is None:
        # Default chunk size based on available memory
        available_memory = get_memory_usage()['available_mb']
        chunk_size = max(1000, int(len(df) * min(0.1, available_memory / 1000)))
    
    elif max_memory_mb is not None:
        # Calculate chunk size based on memory limit
        df_memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        if df_memory_mb <= max_memory_mb:
            chunk_size = len(df)  # Entire DataFrame fits
        else:
            chunk_size = max(1, int(len(df) * (max_memory_mb / df_memory_mb)))
    
    if chunk_size >= len(df):
        yield df
        return
    
    logger.info(f"Chunking DataFrame: {len(df)} rows -> {chunk_size} rows/chunk")
    
    for start_idx in range(0, len(df), chunk_size):
        end_idx = min(start_idx + chunk_size, len(df))
        chunk = df.iloc[start_idx:end_idx].copy()
        
        logger.debug(f"Processing chunk {start_idx//chunk_size + 1}: rows {start_idx}-{end_idx}")
        yield chunk


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to optimize
        
    Returns
    -------
    pd.DataFrame
        Memory-optimized DataFrame
    """
    logger.info("Optimizing DataFrame memory usage")
    initial_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    optimized_df = df.copy()
    
    for col in optimized_df.columns:
        col_type = optimized_df[col].dtype
        
        if col_type == 'int64':
            # Try to downcast integers
            try:
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
            except (ValueError, TypeError):
                pass
        
        elif col_type == 'float64':
            # Try to downcast floats
            try:
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
            except (ValueError, TypeError):
                pass
        
        elif col_type == 'object':
            # Convert strings to categorical if beneficial
            unique_count = optimized_df[col].nunique()
            total_count = len(optimized_df[col])
            
            if unique_count < total_count * 0.5:  # Less than 50% unique values
                try:
                    optimized_df[col] = optimized_df[col].astype('category')
                except (ValueError, TypeError):
                    pass
    
    final_memory = optimized_df.memory_usage(deep=True).sum() / (1024 * 1024)
    memory_reduction = ((initial_memory - final_memory) / initial_memory) * 100
    
    logger.info(
        f"Memory optimization: {initial_memory:.1f}MB -> {final_memory:.1f}MB "
        f"({memory_reduction:.1f}% reduction)"
    )
    
    return optimized_df


def process_large_dataframe(
    df: pd.DataFrame,
    processing_func: Callable[[pd.DataFrame], pd.DataFrame],
    chunk_size: Optional[int] = None,
    max_memory_mb: Optional[float] = None,
    optimize_memory: bool = True
) -> pd.DataFrame:
    """
    Process large DataFrame in chunks to manage memory usage.
    
    Parameters
    ----------
    df : pd.DataFrame
        Large DataFrame to process
    processing_func : Callable
        Function to apply to each chunk
    chunk_size : int, optional
        Number of rows per chunk
    max_memory_mb : float, optional
        Maximum memory per chunk in megabytes
    optimize_memory : bool, default True
        Whether to optimize memory usage
        
    Returns
    -------
    pd.DataFrame
        Processed DataFrame
    """
    logger.info(f"Processing large DataFrame: {len(df)} rows, {len(df.columns)} columns")
    
    if optimize_memory:
        df = optimize_dataframe_memory(df)
    
    # Check if chunking is necessary
    df_memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    available_memory = get_memory_usage()['available_mb']
    
    if df_memory_mb < available_memory * 0.1:  # Less than 10% of available memory
        logger.info("DataFrame small enough for direct processing")
        return processing_func(df)
    
    # Process in chunks
    processed_chunks = []
    
    for i, chunk in enumerate(chunk_dataframe(df, chunk_size, max_memory_mb)):
        logger.debug(f"Processing chunk {i+1}")
        
        with memory_limit_context(max_memory_mb or available_memory * 0.2):
            processed_chunk = processing_func(chunk)
            processed_chunks.append(processed_chunk)
        
        # Force garbage collection between chunks
        gc.collect()
    
    logger.info(f"Combining {len(processed_chunks)} processed chunks")
    result = pd.concat(processed_chunks, ignore_index=True)
    
    if optimize_memory:
        result = optimize_dataframe_memory(result)
    
    return result


def benchmark_operation(
    operation_func: Callable,
    data_sizes: list,
    create_data_func: Callable[[int], pd.DataFrame],
    operation_name: str = "operation"
) -> Dict[str, Any]:
    """
    Benchmark an operation across different data sizes.
    
    Parameters
    ----------
    operation_func : Callable
        Function to benchmark
    data_sizes : list
        List of data sizes to test
    create_data_func : Callable
        Function to create test data of given size
    operation_name : str
        Name of the operation for reporting
        
    Returns
    -------
    Dict[str, Any]
        Benchmark results
    """
    logger.info(f"Benchmarking {operation_name} across {len(data_sizes)} data sizes")
    
    results = {
        'operation': operation_name,
        'data_sizes': data_sizes,
        'durations': [],
        'memory_deltas': [],
        'throughput': []  # rows per second
    }
    
    for size in data_sizes:
        logger.info(f"Benchmarking with {size} rows")
        
        # Create test data
        test_data = create_data_func(size)
        
        # Benchmark operation
        start_time = time.time()
        start_memory = get_memory_usage()
        
        try:
            result = operation_func(test_data)
            
            duration = time.time() - start_time
            end_memory = get_memory_usage()
            memory_delta = end_memory['rss_mb'] - start_memory['rss_mb']
            throughput = size / duration if duration > 0 else 0
            
            results['durations'].append(duration)
            results['memory_deltas'].append(memory_delta)
            results['throughput'].append(throughput)
            
            logger.info(
                f"Size {size}: {duration:.2f}s, "
                f"memory: {memory_delta:+.1f}MB, "
                f"throughput: {throughput:.0f} rows/s"
            )
            
        except Exception as e:
            logger.error(f"Benchmark failed for size {size}: {e}")
            results['durations'].append(None)
            results['memory_deltas'].append(None)
            results['throughput'].append(None)
        
        # Clean up
        del test_data
        if 'result' in locals():
            del result
        gc.collect()
    
    return results


def check_performance_regression(
    current_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
    tolerance_percent: float = 10.0
) -> Dict[str, Any]:
    """
    Check for performance regressions compared to baseline.
    
    Parameters
    ----------
    current_metrics : Dict[str, Any]
        Current performance metrics
    baseline_metrics : Dict[str, Any] 
        Baseline performance metrics
    tolerance_percent : float, default 10.0
        Allowed performance degradation percentage
        
    Returns
    -------
    Dict[str, Any]
        Regression analysis results
    """
    analysis = {
        'has_regression': False,
        'regressions': [],
        'improvements': [],
        'summary': {}
    }
    
    for metric_name in ['duration', 'memory_delta']:
        if metric_name in current_metrics and metric_name in baseline_metrics:
            current_val = current_metrics[metric_name]
            baseline_val = baseline_metrics[metric_name]
            
            if baseline_val > 0:
                change_percent = ((current_val - baseline_val) / baseline_val) * 100
                
                if change_percent > tolerance_percent:
                    analysis['has_regression'] = True
                    analysis['regressions'].append({
                        'metric': metric_name,
                        'change_percent': change_percent,
                        'current': current_val,
                        'baseline': baseline_val
                    })
                elif change_percent < -tolerance_percent:
                    analysis['improvements'].append({
                        'metric': metric_name,
                        'improvement_percent': -change_percent,
                        'current': current_val,
                        'baseline': baseline_val
                    })
    
    analysis['summary'] = {
        'regression_count': len(analysis['regressions']),
        'improvement_count': len(analysis['improvements']),
        'overall_status': 'regression' if analysis['has_regression'] else 'acceptable'
    }
    
    return analysis