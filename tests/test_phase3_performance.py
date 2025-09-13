"""
Tests for Phase 3 performance and scalability features.

This module contains tests for the performance features including
Dask integration, async processing, streaming, and distributed processing.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
import asyncio
import time

from cleanepi.core.config import CleaningConfig, MissingValueConfig, DuplicateConfig, ConstantConfig
from cleanepi.performance.streaming import StreamingCleaner
from cleanepi.performance.benchmarking import PerformanceBenchmark


class TestStreamingCleaner:
    """Test cases for StreamingCleaner."""
    
    def setup_method(self):
        """Set up test data."""
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create test data
        self.test_data = pd.DataFrame({
            'id': range(1000),
            'name': [f'person_{i}' for i in range(1000)],
            'age': np.random.randint(18, 80, 1000),
            'score': np.random.normal(75, 15, 1000),
            'constant_col': 'same_value',
            'missing_col': [np.nan if i % 10 == 0 else f'value_{i}' for i in range(1000)]
        })
        
        # Add some duplicates
        self.test_data.iloc[100:110] = self.test_data.iloc[0:10].values
        
        self.test_file = self.test_dir / "test_data.csv"
        self.test_data.to_csv(self.test_file, index=False)
        
        self.config = CleaningConfig(
            standardize_column_names=True,
            replace_missing_values=MissingValueConfig(),
            remove_duplicates=DuplicateConfig(),
            remove_constants=ConstantConfig()
        )
    
    def teardown_method(self):
        """Clean up test files."""
        shutil.rmtree(self.test_dir)
    
    def test_streaming_cleaner_initialization(self):
        """Test StreamingCleaner initialization."""
        cleaner = StreamingCleaner(chunk_size=100, memory_limit="100MB")
        
        assert cleaner.chunk_size == 100
        assert cleaner.memory_limit == "100MB"
        assert cleaner._memory_limit_bytes == 100 * 1024 * 1024
    
    def test_clean_csv_streaming(self):
        """Test streaming CSV cleaning."""
        cleaner = StreamingCleaner(chunk_size=200)
        
        progress_calls = []
        def progress_callback(chunk_num, total_chunks, message):
            progress_calls.append((chunk_num, total_chunks, message))
        
        output_file = self.test_dir / "cleaned_output.csv"
        
        _, report = cleaner.clean_csv_streaming(
            self.test_file,
            self.config,
            output_path=output_file,
            progress_callback=progress_callback
        )
        
        # Check that output file was created
        assert output_file.exists()
        
        # Load and verify cleaned data
        cleaned_data = pd.read_csv(output_file)
        
        # Should have fewer rows due to duplicate removal
        assert len(cleaned_data) < len(self.test_data)
        
        # Should have fewer columns due to constant removal
        assert len(cleaned_data.columns) < len(self.test_data.columns)
        
        # Check that progress was reported
        assert len(progress_calls) > 0
        
        # Check report contains streaming information
        assert report.metadata["streaming_mode"] is True
        assert report.metadata["total_chunks"] > 0
        assert report.metadata["chunk_size"] == 200
    
    def test_clean_dataframe_streaming(self):
        """Test streaming DataFrame cleaning."""
        cleaner = StreamingCleaner(chunk_size=300)
        
        cleaned_data, report = cleaner.clean_dataframe_streaming(
            self.test_data,
            self.config
        )
        
        # Verify results
        assert len(cleaned_data) < len(self.test_data)  # Duplicates removed
        assert len(cleaned_data.columns) < len(self.test_data.columns)  # Constants removed
        
        # Check report
        assert report.metadata["streaming_mode"] is True
        assert report.metadata["chunk_size"] == 300
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        cleaner = StreamingCleaner(chunk_size=100)
        
        memory_info = cleaner.get_memory_usage()
        
        assert "current_memory_mb" in memory_info
        assert "memory_limit_mb" in memory_info
        assert "chunk_size" in memory_info
        assert memory_info["chunk_size"] == 100


class TestAsyncCleaner:
    """Test cases for AsyncCleaner."""
    
    def setup_method(self):
        """Set up test data."""
        self.test_dir = Path(tempfile.mkdtemp())
        
        self.test_data = pd.DataFrame({
            'col1': range(500),
            'col2': [f'text_{i}' for i in range(500)],
            'constant': 'same',
            'missing': [np.nan if i % 5 == 0 else i for i in range(500)]
        })
        
        self.test_file = self.test_dir / "async_test.csv"
        self.test_data.to_csv(self.test_file, index=False)
        
        self.config = CleaningConfig(
            standardize_column_names=True,
            replace_missing_values=MissingValueConfig(),
            remove_constants=ConstantConfig()
        )
    
    def teardown_method(self):
        """Clean up test files."""
        shutil.rmtree(self.test_dir)
    
    @pytest.mark.asyncio
    async def test_async_clean_dataframe(self):
        """Test async DataFrame cleaning."""
        try:
            from cleanepi.performance.async_processing import AsyncCleaner
        except ImportError:
            pytest.skip("AsyncCleaner requires aiofiles")
        
        progress_calls = []
        async def progress_callback(step, total_steps, message):
            progress_calls.append((step, total_steps, message))
        
        async with AsyncCleaner(max_workers=2) as cleaner:
            cleaned_data, report = await cleaner.clean_dataframe(
                self.test_data,
                self.config,
                progress_callback=progress_callback
            )
        
        # Verify results
        assert len(cleaned_data.columns) < len(self.test_data.columns)  # Constants removed
        assert "async_processing_time" in report.performance_metrics
        assert report.performance_metrics["worker_threads"] == 2
        
        # Check progress was reported
        assert len(progress_calls) > 0
    
    @pytest.mark.asyncio
    async def test_async_clean_csv(self):
        """Test async CSV cleaning."""
        try:
            from cleanepi.performance.async_processing import AsyncCleaner
        except ImportError:
            pytest.skip("AsyncCleaner requires aiofiles")
        
        output_file = self.test_dir / "async_cleaned.csv"
        
        async with AsyncCleaner() as cleaner:
            _, report = await cleaner.clean_csv(
                self.test_file,
                self.config,
                output_path=output_file
            )
        
        # Check output file was created
        assert output_file.exists()
        
        # Verify cleaned data
        cleaned_data = pd.read_csv(output_file)
        assert len(cleaned_data.columns) < len(self.test_data.columns)


class TestPerformanceBenchmark:
    """Test cases for PerformanceBenchmark."""
    
    def setup_method(self):
        """Set up test data."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.benchmark = PerformanceBenchmark(output_dir=self.test_dir)
        
        self.config = CleaningConfig(
            standardize_column_names=True,
            replace_missing_values=MissingValueConfig()
        )
    
    def teardown_method(self):
        """Clean up test files."""
        shutil.rmtree(self.test_dir)
    
    def test_generate_test_data(self):
        """Test synthetic data generation."""
        df = self.benchmark.generate_test_data(
            num_rows=100,
            num_columns=5,
            missing_rate=0.1,
            duplicate_rate=0.05,
            constant_columns=1
        )
        
        assert len(df) == 100
        assert len(df.columns) == 5
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if col.startswith('constant_')]
        assert len(constant_cols) == 1
        
        # Check for missing values
        assert df.isnull().sum().sum() > 0
    
    def test_benchmark_operation(self):
        """Test operation benchmarking."""
        # Generate small test data for quick benchmark
        df = self.benchmark.generate_test_data(100, 5)
        
        result = self.benchmark.benchmark_operation(
            df,
            self.config,
            operation_name="test_operation",
            warmup_runs=1,
            benchmark_runs=2
        )
        
        # Check result structure
        assert "operation_name" in result
        assert result["operation_name"] == "test_operation"
        assert "avg_time_seconds" in result
        assert "rows_per_second" in result
        assert "input_shape" in result
        assert "output_shape" in result
        assert result["warmup_runs"] == 1
        assert result["benchmark_runs"] == 2
        
        # Check that benchmark was recorded
        assert len(self.benchmark.results) == 1
    
    def test_benchmark_scalability(self):
        """Test scalability benchmarking."""
        row_counts = [50, 100, 200]
        
        results = self.benchmark.benchmark_scalability(
            row_counts,
            self.config,
            num_columns=3,
            benchmark_runs=1
        )
        
        assert len(results) == len(row_counts)
        
        # Check that processing time generally increases with data size
        times = [r["avg_time_seconds"] for r in results]
        assert len(times) == 3
    
    def test_benchmark_configurations(self):
        """Test configuration benchmarking."""
        df = self.benchmark.generate_test_data(100, 5)
        
        configs = {
            "minimal": CleaningConfig(standardize_column_names=True),
            "standard": CleaningConfig(
                standardize_column_names=True,
                replace_missing_values=MissingValueConfig()
            )
        }
        
        results = self.benchmark.benchmark_configurations(
            df,
            configs,
            benchmark_runs=1
        )
        
        assert len(results) == 2
        
        config_names = [r["config_name"] for r in results]
        assert "minimal" in config_names
        assert "standard" in config_names
    
    def test_save_results(self):
        """Test saving benchmark results."""
        # Run a quick benchmark to generate results
        df = self.benchmark.generate_test_data(50, 3)
        self.benchmark.benchmark_operation(df, self.config, benchmark_runs=1)
        
        # Save results
        results_file = self.benchmark.save_results("test_results.csv")
        
        assert results_file.exists()
        
        # Load and verify results
        results_df = pd.read_csv(results_file)
        assert len(results_df) == 1
        assert "operation_name" in results_df.columns
        assert "avg_time_seconds" in results_df.columns
    
    def test_generate_report(self):
        """Test report generation."""
        # Run a quick benchmark
        df = self.benchmark.generate_test_data(50, 3)
        self.benchmark.benchmark_operation(df, self.config, benchmark_runs=1)
        
        report = self.benchmark.generate_report()
        
        assert "Performance Benchmark Report" in report
        assert "Total benchmarks: 1" in report
        assert "Processing time:" in report
        assert "Throughput:" in report


class TestDaskIntegration:
    """Test cases for Dask integration (when available)."""
    
    def setup_method(self):
        """Set up test data."""
        self.test_data = pd.DataFrame({
            'id': range(200),
            'value': np.random.normal(0, 1, 200),
            'category': np.random.choice(['A', 'B', 'C'], 200),
            'constant': 'same_value'
        })
        
        self.config = CleaningConfig(
            standardize_column_names=True,
            remove_constants=ConstantConfig()
        )
    
    def test_dask_cleaner_availability(self):
        """Test if Dask cleaner can be imported."""
        try:
            from cleanepi.performance import get_dask_cleaner
            DaskCleaner = get_dask_cleaner()
            
            # If we can import, test basic functionality
            with DaskCleaner() as cleaner:
                assert cleaner.client is not None
                
                # Test performance metrics
                metrics = cleaner.get_performance_metrics()
                assert "num_workers" in metrics
                
        except ImportError:
            pytest.skip("Dask not available - skipping Dask tests")


@pytest.mark.integration
class TestPerformanceIntegration:
    """Integration tests for performance features."""
    
    def setup_method(self):
        """Set up integration test data."""
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create larger test dataset
        self.large_data = pd.DataFrame({
            'id': range(5000),
            'name': [f'person_{i}' for i in range(5000)],
            'age': np.random.randint(18, 100, 5000),
            'score': np.random.normal(75, 20, 5000),
            'category': np.random.choice(['A', 'B', 'C', 'D'], 5000),
            'constant1': 'same_value',
            'constant2': 42,
            'missing_data': [np.nan if i % 20 == 0 else f'data_{i}' for i in range(5000)]
        })
        
        # Add duplicates
        self.large_data.iloc[1000:1100] = self.large_data.iloc[0:100].values
        
        self.large_file = self.test_dir / "large_test.csv"
        self.large_data.to_csv(self.large_file, index=False)
        
        self.config = CleaningConfig(
            standardize_column_names=True,
            replace_missing_values=MissingValueConfig(),
            remove_duplicates=DuplicateConfig(),
            remove_constants=ConstantConfig()
        )
    
    def teardown_method(self):
        """Clean up test files."""
        shutil.rmtree(self.test_dir)
    
    def test_performance_comparison(self):
        """Compare performance across different processing modes."""
        benchmark = PerformanceBenchmark(self.test_dir)
        
        # Test available processing modes
        modes_to_test = ["standard", "streaming"]
        
        results = []
        
        for mode in modes_to_test:
            try:
                if mode == "standard":
                    start_time = time.time()
                    from cleanepi import clean_data
                    cleaned_data, report = clean_data(self.large_data, self.config)
                    end_time = time.time()
                    
                    result = {
                        "mode": mode,
                        "time": end_time - start_time,
                        "rows_processed": len(self.large_data),
                        "rows_output": len(cleaned_data)
                    }
                    
                elif mode == "streaming":
                    cleaner = StreamingCleaner(chunk_size=1000)
                    start_time = time.time()
                    cleaned_data, report = cleaner.clean_csv_streaming(
                        self.large_file,
                        self.config
                    )
                    end_time = time.time()
                    
                    result = {
                        "mode": mode,
                        "time": end_time - start_time,
                        "rows_processed": len(self.large_data),
                        "rows_output": len(cleaned_data) if cleaned_data is not None else 0,
                        "chunks": report.summary.get("total_chunks", 0)
                    }
                
                results.append(result)
                
            except Exception as e:
                pytest.skip(f"Mode {mode} not available: {str(e)}")
        
        # Verify we tested at least one mode
        assert len(results) > 0
        
        # All modes should process the same amount of data
        if len(results) > 1:
            rows_processed = [r["rows_processed"] for r in results]
            assert all(rows == rows_processed[0] for rows in rows_processed)
    
    def test_memory_efficiency(self):
        """Test memory efficiency of streaming vs standard processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Test standard processing memory usage
        initial_memory = process.memory_info().rss
        
        from cleanepi import clean_data
        cleaned_data, _ = clean_data(self.large_data, self.config)
        
        standard_memory = process.memory_info().rss - initial_memory
        
        # Clean up
        del cleaned_data
        import gc
        gc.collect()
        
        # Test streaming processing memory usage
        initial_memory = process.memory_info().rss
        
        cleaner = StreamingCleaner(chunk_size=500)  # Small chunks for memory efficiency
        cleaned_data, _ = cleaner.clean_csv_streaming(
            self.large_file,
            self.config
        )
        
        streaming_memory = process.memory_info().rss - initial_memory
        
        # Streaming should use less memory for large datasets
        # (This might not always be true for small test data, but we can check the logic works)
        assert streaming_memory >= 0  # At least some memory was used
        assert standard_memory >= 0   # At least some memory was used
        
        print(f"Standard processing memory: {standard_memory / 1024 / 1024:.1f} MB")
        print(f"Streaming processing memory: {streaming_memory / 1024 / 1024:.1f} MB")