"""
Performance benchmarking and optimization tools for cleanepi.

This module provides benchmarking utilities to measure and optimize
the performance of data cleaning operations across different configurations.
"""

import time
import psutil
import gc
import os
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

from ..core.config import CleaningConfig
from ..core.clean_data import clean_data
from ..core.report import CleaningReport


class PerformanceBenchmark:
    """
    Performance benchmarking suite for cleanepi operations.
    
    This class provides comprehensive benchmarking capabilities to measure
    performance across different data sizes, configurations, and processing modes.
    """
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize PerformanceBenchmark.
        
        Parameters
        ----------
        output_dir : str or Path, optional
            Directory to save benchmark results. If None, uses current directory.
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
        logger.info(f"PerformanceBenchmark initialized, output dir: {self.output_dir}")
    
    def generate_test_data(
        self,
        num_rows: int,
        num_columns: int = 10,
        missing_rate: float = 0.1,
        duplicate_rate: float = 0.05,
        constant_columns: int = 2,
        data_types: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic test data for benchmarking.
        
        Parameters
        ----------
        num_rows : int
            Number of rows to generate.
        num_columns : int, default 10
            Number of columns to generate.
        missing_rate : float, default 0.1
            Proportion of missing values to introduce.
        duplicate_rate : float, default 0.05
            Proportion of duplicate rows to introduce.
        constant_columns : int, default 2
            Number of constant columns to include.
        data_types : dict, optional
            Dictionary specifying data types for columns.
            
        Returns
        -------
        pd.DataFrame
            Generated test DataFrame.
        """
        logger.info(f"Generating test data: {num_rows} rows, {num_columns} columns")
        
        np.random.seed(42)  # For reproducible results
        
        data = {}
        
        # Generate different types of columns
        for i in range(num_columns - constant_columns):
            col_name = f"column_{i+1}"
            
            if data_types and col_name in data_types:
                dtype = data_types[col_name]
            else:
                dtype = np.random.choice(['int', 'float', 'string', 'date'])
            
            if dtype == 'int':
                data[col_name] = np.random.randint(0, 1000, num_rows)
            elif dtype == 'float':
                data[col_name] = np.random.normal(100, 20, num_rows)
            elif dtype == 'string':
                categories = [f'cat_{j}' for j in range(20)]
                data[col_name] = np.random.choice(categories, num_rows)
            elif dtype == 'date':
                start_date = pd.Timestamp('2020-01-01')
                data[col_name] = pd.date_range(
                    start=start_date,
                    periods=num_rows,
                    freq='D'
                )
        
        # Add constant columns
        for i in range(constant_columns):
            data[f"constant_{i+1}"] = 'constant_value'
        
        df = pd.DataFrame(data)
        
        # Introduce missing values
        if missing_rate > 0:
            missing_mask = np.random.random(df.shape) < missing_rate
            df = df.mask(missing_mask)
        
        # Introduce duplicates
        if duplicate_rate > 0:
            num_duplicates = int(num_rows * duplicate_rate)
            duplicate_indices = np.random.choice(
                num_rows - num_duplicates,
                num_duplicates,
                replace=False
            )
            for idx in duplicate_indices:
                df.iloc[num_rows - num_duplicates + np.where(duplicate_indices == idx)[0][0]] = df.iloc[idx]
        
        logger.info(f"Generated test data with shape: {df.shape}")
        return df
    
    def benchmark_operation(
        self,
        df: pd.DataFrame,
        config: CleaningConfig,
        operation_name: str = "full_cleaning",
        warmup_runs: int = 1,
        benchmark_runs: int = 3
    ) -> Dict[str, Any]:
        """
        Benchmark a specific cleaning operation.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to benchmark.
        config : CleaningConfig
            Configuration for cleaning operations.
        operation_name : str, default "full_cleaning"
            Name of the operation being benchmarked.
        warmup_runs : int, default 1
            Number of warmup runs before benchmarking.
        benchmark_runs : int, default 3
            Number of benchmark runs to average.
            
        Returns
        -------
        dict
            Dictionary containing benchmark results.
        """
        logger.info(f"Benchmarking operation: {operation_name}")
        
        # Warmup runs
        for _ in range(warmup_runs):
            _, _ = clean_data(df.copy(), config)
            gc.collect()
        
        # Benchmark runs
        times = []
        memory_usages = []
        results = []
        
        for run in range(benchmark_runs):
            # Get initial memory
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
            
            # Time the operation
            start_time = time.perf_counter()
            cleaned_df, report = clean_data(df.copy(), config)
            end_time = time.perf_counter()
            
            # Get peak memory
            peak_memory = process.memory_info().rss
            memory_usage = peak_memory - initial_memory
            
            times.append(end_time - start_time)
            memory_usages.append(memory_usage)
            results.append((cleaned_df, report))
            
            gc.collect()
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        avg_memory = np.mean(memory_usages)
        std_memory = np.std(memory_usages)
        
        # Get final result metrics
        final_df, final_report = results[0]
        
        benchmark_result = {
            "operation_name": operation_name,
            "input_shape": df.shape,
            "output_shape": final_df.shape,
            "input_memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "output_memory_mb": final_df.memory_usage(deep=True).sum() / 1024 / 1024,
            "avg_time_seconds": avg_time,
            "std_time_seconds": std_time,
            "min_time_seconds": min_time,
            "max_time_seconds": max_time,
            "avg_memory_mb": avg_memory / 1024 / 1024,
            "std_memory_mb": std_memory / 1024 / 1024,
            "rows_per_second": len(df) / avg_time,
            "memory_reduction_ratio": (
                (df.memory_usage(deep=True).sum() - final_df.memory_usage(deep=True).sum()) /
                df.memory_usage(deep=True).sum()
            ),
            "throughput_mb_per_second": (df.memory_usage(deep=True).sum() / 1024 / 1024) / avg_time,
            "operations_performed": [op.operation for op in final_report.operations],
            "warmup_runs": warmup_runs,
            "benchmark_runs": benchmark_runs,
            "timestamp": time.time()
        }
        
        self.results.append(benchmark_result)
        
        logger.info(
            f"Benchmark completed: {avg_time:.3f}s avg, "
            f"{benchmark_result['rows_per_second']:.0f} rows/s, "
            f"{avg_memory / 1024 / 1024:.1f}MB memory"
        )
        
        return benchmark_result
    
    def benchmark_scalability(
        self,
        row_counts: List[int],
        config: CleaningConfig,
        num_columns: int = 10,
        benchmark_runs: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Benchmark scalability across different data sizes.
        
        Parameters
        ----------
        row_counts : list
            List of row counts to benchmark.
        config : CleaningConfig
            Configuration for cleaning operations.
        num_columns : int, default 10
            Number of columns in test data.
        benchmark_runs : int, default 3
            Number of benchmark runs per size.
            
        Returns
        -------
        list
            List of benchmark results for each size.
        """
        logger.info(f"Starting scalability benchmark: {row_counts}")
        
        scalability_results = []
        
        for row_count in row_counts:
            logger.info(f"Benchmarking with {row_count} rows")
            
            # Generate test data
            df = self.generate_test_data(row_count, num_columns)
            
            # Benchmark
            result = self.benchmark_operation(
                df,
                config,
                operation_name=f"scalability_{row_count}_rows",
                benchmark_runs=benchmark_runs
            )
            
            scalability_results.append(result)
            
            # Clean up
            del df
            gc.collect()
        
        return scalability_results
    
    def benchmark_configurations(
        self,
        df: pd.DataFrame,
        configs: Dict[str, CleaningConfig],
        benchmark_runs: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Benchmark different cleaning configurations.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to benchmark.
        configs : dict
            Dictionary mapping configuration names to CleaningConfig objects.
        benchmark_runs : int, default 3
            Number of benchmark runs per configuration.
            
        Returns
        -------
        list
            List of benchmark results for each configuration.
        """
        logger.info(f"Benchmarking {len(configs)} configurations")
        
        config_results = []
        
        for config_name, config in configs.items():
            logger.info(f"Benchmarking configuration: {config_name}")
            
            result = self.benchmark_operation(
                df,
                config,
                operation_name=f"config_{config_name}",
                benchmark_runs=benchmark_runs
            )
            
            result["config_name"] = config_name
            config_results.append(result)
        
        return config_results
    
    def benchmark_processing_modes(
        self,
        df: pd.DataFrame,
        config: CleaningConfig,
        modes: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Benchmark different processing modes (standard, streaming, async, distributed).
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to benchmark.
        config : CleaningConfig
            Configuration for cleaning operations.
        modes : list, optional
            List of processing modes to benchmark. If None, benchmarks all available.
            
        Returns
        -------
        list
            List of benchmark results for each processing mode.
        """
        if modes is None:
            modes = ["standard", "streaming"]
            
            # Check for optional dependencies
            try:
                from ..performance import get_async_cleaner
                modes.append("async")
            except ImportError:
                pass
            
            try:
                from ..performance import get_dask_cleaner
                modes.append("dask")
            except ImportError:
                pass
        
        logger.info(f"Benchmarking processing modes: {modes}")
        
        mode_results = []
        
        for mode in modes:
            logger.info(f"Benchmarking mode: {mode}")
            
            try:
                if mode == "standard":
                    result = self.benchmark_operation(df, config, f"mode_{mode}")
                
                elif mode == "streaming":
                    result = self._benchmark_streaming(df, config)
                
                elif mode == "async":
                    result = self._benchmark_async(df, config)
                
                elif mode == "dask":
                    result = self._benchmark_dask(df, config)
                
                else:
                    logger.warning(f"Unknown processing mode: {mode}")
                    continue
                
                result["processing_mode"] = mode
                mode_results.append(result)
                
            except Exception as e:
                logger.error(f"Error benchmarking mode {mode}: {str(e)}")
                
                # Create error result
                error_result = {
                    "operation_name": f"mode_{mode}",
                    "processing_mode": mode,
                    "error": str(e),
                    "timestamp": time.time()
                }
                mode_results.append(error_result)
        
        return mode_results
    
    def _benchmark_streaming(self, df: pd.DataFrame, config: CleaningConfig) -> Dict[str, Any]:
        """Benchmark streaming processing mode."""
        from ..performance.streaming import StreamingCleaner
        
        start_time = time.perf_counter()
        
        streaming_cleaner = StreamingCleaner(chunk_size=5000)
        cleaned_df, report = streaming_cleaner.clean_dataframe_streaming(df, config)
        
        end_time = time.perf_counter()
        
        return {
            "operation_name": "mode_streaming",
            "input_shape": df.shape,
            "output_shape": cleaned_df.shape,
            "avg_time_seconds": end_time - start_time,
            "rows_per_second": len(df) / (end_time - start_time),
            "streaming_chunks": report.metadata.get("total_chunks", 0),
            "chunk_size": 5000,
            "timestamp": time.time()
        }
    
    def _benchmark_async(self, df: pd.DataFrame, config: CleaningConfig) -> Dict[str, Any]:
        """Benchmark async processing mode."""
        import asyncio
        from ..performance.async_processing import AsyncCleaner
        
        async def run_async_benchmark():
            start_time = time.perf_counter()
            
            async with AsyncCleaner(max_workers=2) as async_cleaner:
                cleaned_df, report = await async_cleaner.clean_dataframe(df, config)
            
            end_time = time.perf_counter()
            
            return {
                "operation_name": "mode_async",
                "input_shape": df.shape,
                "output_shape": cleaned_df.shape,
                "avg_time_seconds": end_time - start_time,
                "rows_per_second": len(df) / (end_time - start_time),
                "max_workers": 2,
                "timestamp": time.time()
            }
        
        # Run the async benchmark
        return asyncio.run(run_async_benchmark())
    
    def _benchmark_dask(self, df: pd.DataFrame, config: CleaningConfig) -> Dict[str, Any]:
        """Benchmark Dask processing mode."""
        try:
            import dask.dataframe as dd
            from ..performance.dask_processing import DaskCleaner
        except ImportError:
            raise ImportError("Dask benchmarking requires dask to be installed")
        
        start_time = time.perf_counter()
        
        with DaskCleaner() as dask_cleaner:
            # Convert to Dask DataFrame
            ddf = dd.from_pandas(df, npartitions=4)
            
            # Clean with Dask
            cleaned_ddf, report = dask_cleaner.clean_dataframe(ddf, config)
            
            # Compute result
            cleaned_df = cleaned_ddf.compute()
        
        end_time = time.perf_counter()
        
        return {
            "operation_name": "mode_dask",
            "input_shape": df.shape,
            "output_shape": cleaned_df.shape,
            "avg_time_seconds": end_time - start_time,
            "rows_per_second": len(df) / (end_time - start_time),
            "partitions": 4,
            "timestamp": time.time()
        }
    
    def save_results(self, filename: Optional[str] = None) -> Path:
        """
        Save benchmark results to CSV file.
        
        Parameters
        ----------
        filename : str, optional
            Name of output file. If None, generates timestamp-based name.
            
        Returns
        -------
        Path
            Path to saved results file.
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.csv"
        
        output_path = self.output_dir / filename
        
        if self.results:
            results_df = pd.DataFrame(self.results)
            results_df.to_csv(output_path, index=False)
            logger.info(f"Benchmark results saved to: {output_path}")
        else:
            logger.warning("No benchmark results to save")
        
        return output_path
    
    def generate_report(self) -> str:
        """
        Generate a text report of benchmark results.
        
        Returns
        -------
        str
            Formatted benchmark report.
        """
        if not self.results:
            return "No benchmark results available."
        
        report_lines = [
            "=== Performance Benchmark Report ===",
            f"Total benchmarks: {len(self.results)}",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Group results by operation type
        operations = {}
        for result in self.results:
            op_name = result.get("operation_name", "unknown")
            if op_name not in operations:
                operations[op_name] = []
            operations[op_name].append(result)
        
        for op_name, op_results in operations.items():
            report_lines.extend([
                f"Operation: {op_name}",
                "-" * (len(op_name) + 11)
            ])
            
            for result in op_results:
                if "error" in result:
                    report_lines.append(f"  ERROR: {result['error']}")
                    continue
                
                report_lines.extend([
                    f"  Input shape: {result.get('input_shape', 'N/A')}",
                    f"  Output shape: {result.get('output_shape', 'N/A')}",
                    f"  Processing time: {result.get('avg_time_seconds', 0):.3f} seconds",
                    f"  Throughput: {result.get('rows_per_second', 0):.0f} rows/second",
                    f"  Memory usage: {result.get('avg_memory_mb', 0):.1f} MB",
                    ""
                ])
        
        return "\n".join(report_lines)
    
    def clear_results(self):
        """Clear all benchmark results."""
        self.results.clear()
        logger.info("Benchmark results cleared")


def run_comprehensive_benchmark(
    output_dir: Optional[Union[str, Path]] = None,
    max_rows: int = 100000
) -> Path:
    """
    Run a comprehensive performance benchmark suite.
    
    Parameters
    ----------
    output_dir : str or Path, optional
        Directory to save results.
    max_rows : int, default 100000
        Maximum number of rows for scalability testing.
        
    Returns
    -------
    Path
        Path to saved benchmark results.
    """
    logger.info("Starting comprehensive performance benchmark")
    
    benchmark = PerformanceBenchmark(output_dir)
    
    # Define test configurations
    configs = {
        "minimal": CleaningConfig(
            standardize_column_names=True
        ),
        "standard": CleaningConfig(
            standardize_column_names=True,
            replace_missing_values=True,
            remove_duplicates=True
        ),
        "comprehensive": CleaningConfig(
            standardize_column_names=True,
            replace_missing_values=True,
            remove_duplicates=True,
            remove_constants=True
        )
    }
    
    # Scalability benchmark
    row_counts = [1000, 5000, 10000, 25000, 50000]
    if max_rows > 50000:
        row_counts.append(max_rows)
    
    logger.info("Running scalability benchmark...")
    benchmark.benchmark_scalability(
        row_counts,
        configs["standard"],
        num_columns=10,
        benchmark_runs=3
    )
    
    # Configuration benchmark
    logger.info("Running configuration benchmark...")
    test_df = benchmark.generate_test_data(10000, 15)
    benchmark.benchmark_configurations(test_df, configs)
    
    # Processing mode benchmark
    logger.info("Running processing mode benchmark...")
    benchmark.benchmark_processing_modes(test_df, configs["standard"])
    
    # Save results
    results_path = benchmark.save_results()
    
    # Generate and save report
    report = benchmark.generate_report()
    report_path = benchmark.output_dir / "benchmark_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Comprehensive benchmark completed. Results: {results_path}")
    logger.info(f"Report: {report_path}")
    
    return results_path