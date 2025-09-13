"""
Phase 3 Examples: Performance and Scale Features

This script demonstrates the advanced performance and scalability features
introduced in Phase 3 of cleanepi-python.
"""

import pandas as pd
import numpy as np
import time
import asyncio
from pathlib import Path
import tempfile
import shutil

# Import core cleanepi functionality
from cleanepi import clean_data, CleaningConfig
from cleanepi.core.config import MissingValueConfig, DuplicateConfig, ConstantConfig

print("=" * 60)
print("CLEANEPI PHASE 3: PERFORMANCE & SCALE DEMONSTRATION")
print("=" * 60)
print()

# Create a temporary directory for our examples
temp_dir = Path(tempfile.mkdtemp())
print(f"Using temporary directory: {temp_dir}")


def generate_large_dataset(num_rows: int = 50000, num_columns: int = 15) -> pd.DataFrame:
    """Generate a large synthetic dataset for performance testing."""
    print(f"Generating synthetic dataset: {num_rows:,} rows, {num_columns} columns...")
    
    np.random.seed(42)  # For reproducible results
    
    data = {}
    
    # Generate various types of columns
    data['patient_id'] = [f'P{i:06d}' for i in range(num_rows)]
    data['visit_date'] = pd.date_range('2020-01-01', periods=num_rows, freq='H')
    data['age'] = np.random.randint(0, 100, num_rows)
    data['weight_kg'] = np.random.normal(70, 15, num_rows)
    data['height_cm'] = np.random.normal(170, 20, num_rows)
    data['blood_pressure_sys'] = np.random.normal(120, 20, num_rows)
    data['blood_pressure_dia'] = np.random.normal(80, 15, num_rows)
    data['temperature_c'] = np.random.normal(36.5, 1, num_rows)
    data['heart_rate'] = np.random.normal(72, 12, num_rows)
    
    # Add categorical data
    categories = ['Type_A', 'Type_B', 'Type_C', 'Type_D']
    data['diagnosis_category'] = np.random.choice(categories, num_rows)
    
    # Add some constant columns
    data['constant_field_1'] = 'HOSPITAL_A'
    data['constant_field_2'] = 'VERSION_1.0'
    
    # Add columns with missing values
    data['optional_notes'] = [np.nan if i % 10 == 0 else f'note_{i}' for i in range(num_rows)]
    data['lab_result'] = [np.nan if i % 5 == 0 else np.random.normal(50, 10) for i in range(num_rows)]
    
    # Add remaining columns if needed
    remaining_cols = num_columns - len(data)
    for i in range(remaining_cols):
        data[f'extra_col_{i+1}'] = np.random.normal(0, 1, num_rows)
    
    df = pd.DataFrame(data)
    
    # Introduce some duplicates
    num_duplicates = int(num_rows * 0.02)  # 2% duplicates
    duplicate_indices = np.random.choice(num_rows - num_duplicates, num_duplicates, replace=False)
    for i, idx in enumerate(duplicate_indices):
        df.iloc[num_rows - num_duplicates + i] = df.iloc[idx]
    
    print(f"Dataset generated: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    return df


# Generate test datasets of different sizes
datasets = {
    "small": generate_large_dataset(1000, 10),
    "medium": generate_large_dataset(10000, 12), 
    "large": generate_large_dataset(50000, 15)
}

# Save datasets to files
dataset_files = {}
for name, df in datasets.items():
    file_path = temp_dir / f"{name}_dataset.csv"
    df.to_csv(file_path, index=False)
    dataset_files[name] = file_path
    print(f"Saved {name} dataset to: {file_path}")

print()

# Configure cleaning operations
config = CleaningConfig(
    standardize_column_names=True,
    replace_missing_values=MissingValueConfig(),
    remove_duplicates=DuplicateConfig(),
    remove_constants=ConstantConfig()
)

print("Cleaning configuration:")
print(f"  - Standardize column names: {config.standardize_column_names}")
print(f"  - Replace missing values: {config.replace_missing_values is not None}")
print(f"  - Remove duplicates: {config.remove_duplicates is not None}")
print(f"  - Remove constants: {config.remove_constants is not None}")
print()


def demonstrate_standard_processing():
    """Demonstrate standard (in-memory) processing."""
    print("1. STANDARD PROCESSING")
    print("-" * 40)
    
    for size_name, df in datasets.items():
        print(f"\nProcessing {size_name} dataset ({len(df):,} rows)...")
        
        start_time = time.time()
        cleaned_df, report = clean_data(df, config)
        end_time = time.time()
        
        processing_time = end_time - start_time
        rows_per_second = len(df) / processing_time if processing_time > 0 else 0
        
        print(f"  Original shape: {df.shape}")
        print(f"  Cleaned shape:  {cleaned_df.shape}")
        print(f"  Processing time: {processing_time:.3f} seconds")
        print(f"  Throughput: {rows_per_second:,.0f} rows/second")
        print(f"  Memory reduction: {((df.memory_usage(deep=True).sum() - cleaned_df.memory_usage(deep=True).sum()) / df.memory_usage(deep=True).sum() * 100):.1f}%")


def demonstrate_streaming_processing():
    """Demonstrate streaming processing for memory efficiency."""
    print("\n\n2. STREAMING PROCESSING")
    print("-" * 40)
    
    try:
        from cleanepi.performance.streaming import StreamingCleaner
        
        for size_name, file_path in dataset_files.items():
            print(f"\nStreaming {size_name} dataset...")
            
            # Configure streaming cleaner with appropriate chunk size
            chunk_size = 1000 if size_name == "small" else 5000 if size_name == "medium" else 10000
            
            cleaner = StreamingCleaner(
                chunk_size=chunk_size,
                memory_limit="500MB"
            )
            
            progress_updates = []
            def progress_callback(chunk_num, total_chunks, message):
                progress_updates.append((chunk_num, total_chunks, message))
                if total_chunks > 0:
                    percent = (chunk_num / total_chunks) * 100
                    print(f"    Progress: {percent:.1f}% - {message}")
                else:
                    print(f"    Progress: {chunk_num} chunks - {message}")
            
            start_time = time.time()
            
            output_path = temp_dir / f"{size_name}_cleaned_streaming.csv"
            _, report = cleaner.clean_csv_streaming(
                file_path,
                config,
                output_path=output_path,
                progress_callback=progress_callback
            )
            
            end_time = time.time()
            
            # Load cleaned data to get final stats
            cleaned_df = pd.read_csv(output_path)
            original_rows = len(datasets[size_name])
            
            processing_time = end_time - start_time
            rows_per_second = original_rows / processing_time if processing_time > 0 else 0
            
            print(f"  Original rows: {original_rows:,}")
            print(f"  Cleaned rows:  {len(cleaned_df):,}")
            print(f"  Processing time: {processing_time:.3f} seconds")
            print(f"  Throughput: {rows_per_second:,.0f} rows/second")
            print(f"  Chunks processed: {report.summary['total_chunks']}")
            print(f"  Chunk size: {chunk_size:,}")
            
            # Memory usage info
            memory_info = cleaner.get_memory_usage()
            print(f"  Peak memory: {memory_info['current_memory_mb']:.1f} MB")
    
    except ImportError:
        print("Streaming processing not available (install performance extras)")


async def demonstrate_async_processing():
    """Demonstrate asynchronous processing."""
    print("\n\n3. ASYNCHRONOUS PROCESSING")
    print("-" * 40)
    
    try:
        from cleanepi.performance.async_processing import AsyncCleaner
        
        # Test with medium dataset
        size_name = "medium"
        file_path = dataset_files[size_name]
        
        print(f"\nAsync processing {size_name} dataset...")
        
        progress_updates = []
        async def progress_callback(step, total_steps, message):
            progress_updates.append((step, total_steps, message))
            print(f"    {message} ({step}/{total_steps})")
        
        start_time = time.time()
        
        async with AsyncCleaner(max_workers=4) as async_cleaner:
            output_path = temp_dir / f"{size_name}_cleaned_async.csv"
            _, report = await async_cleaner.clean_csv(
                file_path,
                config,
                output_path=output_path,
                progress_callback=progress_callback
            )
        
        end_time = time.time()
        
        # Load results
        cleaned_df = pd.read_csv(output_path)
        original_rows = len(datasets[size_name])
        
        processing_time = end_time - start_time
        rows_per_second = original_rows / processing_time if processing_time > 0 else 0
        
        print(f"  Original rows: {original_rows:,}")
        print(f"  Cleaned rows:  {len(cleaned_df):,}")
        print(f"  Processing time: {processing_time:.3f} seconds")
        print(f"  Throughput: {rows_per_second:,.0f} rows/second")
        print(f"  Worker threads: {4}")
        print(f"  Async overhead: {report.performance_metrics.get('async_processing_time', 0):.3f}s")
    
    except ImportError:
        print("Async processing not available (install async extras)")


def demonstrate_dask_processing():
    """Demonstrate Dask distributed processing."""
    print("\n\n4. DASK DISTRIBUTED PROCESSING")
    print("-" * 40)
    
    try:
        from cleanepi.performance.dask_processing import DaskCleaner
        import dask.dataframe as dd
        
        # Test with large dataset
        size_name = "large"
        df = datasets[size_name]
        
        print(f"\nDask processing {size_name} dataset...")
        
        start_time = time.time()
        
        with DaskCleaner(memory_limit="1GB") as dask_cleaner:
            # Convert to Dask DataFrame
            ddf = dd.from_pandas(df, npartitions=8)
            print(f"  Created Dask DataFrame with {ddf.npartitions} partitions")
            
            # Process with Dask
            cleaned_ddf, report = dask_cleaner.clean_dataframe(ddf, config)
            
            # Compute final result
            cleaned_df = cleaned_ddf.compute()
        
        end_time = time.time()
        
        processing_time = end_time - start_time
        rows_per_second = len(df) / processing_time if processing_time > 0 else 0
        
        print(f"  Original rows: {len(df):,}")
        print(f"  Cleaned rows:  {len(cleaned_df):,}")
        print(f"  Processing time: {processing_time:.3f} seconds")
        print(f"  Throughput: {rows_per_second:,.0f} rows/second")
        print(f"  Partitions: {ddf.npartitions}")
        
        # Get cluster performance metrics
        metrics = dask_cleaner.get_performance_metrics()
        print(f"  Workers used: {metrics.get('num_workers', 'N/A')}")
        print(f"  Total cores: {metrics.get('total_cores', 'N/A')}")
        print(f"  Memory utilization: {metrics.get('memory_utilization', 0) * 100:.1f}%")
    
    except ImportError:
        print("Dask processing not available (install performance extras)")


def demonstrate_benchmarking():
    """Demonstrate performance benchmarking capabilities."""
    print("\n\n5. PERFORMANCE BENCHMARKING")
    print("-" * 40)
    
    try:
        from cleanepi.performance.benchmarking import PerformanceBenchmark
        
        benchmark = PerformanceBenchmark(output_dir=temp_dir)
        
        print("\nRunning benchmarks...")
        
        # Benchmark different data sizes
        print("\n  Scalability benchmark:")
        row_counts = [1000, 5000, 10000]
        scalability_results = benchmark.benchmark_scalability(
            row_counts,
            config,
            num_columns=8,
            benchmark_runs=2
        )
        
        for result in scalability_results:
            rows = result["input_shape"][0]
            time_sec = result["avg_time_seconds"]
            throughput = result["rows_per_second"]
            print(f"    {rows:,} rows: {time_sec:.3f}s ({throughput:,.0f} rows/sec)")
        
        # Benchmark different configurations
        print("\n  Configuration benchmark:")
        test_df = benchmark.generate_test_data(5000, 10)
        
        configs = {
            "minimal": CleaningConfig(standardize_column_names=True),
            "standard": CleaningConfig(
                standardize_column_names=True,
                replace_missing_values=MissingValueConfig(),
                remove_duplicates=DuplicateConfig()
            ),
            "comprehensive": config
        }
        
        config_results = benchmark.benchmark_configurations(test_df, configs, benchmark_runs=2)
        
        for result in config_results:
            config_name = result["config_name"]
            time_sec = result["avg_time_seconds"]
            throughput = result["rows_per_second"]
            print(f"    {config_name}: {time_sec:.3f}s ({throughput:,.0f} rows/sec)")
        
        # Save benchmark results
        results_file = benchmark.save_results("phase3_benchmark_results.csv")
        print(f"\n  Benchmark results saved to: {results_file}")
        
        # Generate report
        report = benchmark.generate_report()
        report_file = temp_dir / "benchmark_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"  Benchmark report saved to: {report_file}")
    
    except ImportError:
        print("Benchmarking not available")


def demonstrate_processing_modes_comparison():
    """Compare different processing modes side by side."""
    print("\n\n6. PROCESSING MODES COMPARISON")
    print("-" * 40)
    
    # Use medium dataset for comparison
    test_df = datasets["medium"]
    test_file = dataset_files["medium"]
    
    results = []
    
    # Standard processing
    print("\nComparing processing modes on medium dataset...")
    
    start_time = time.time()
    cleaned_df, _ = clean_data(test_df, config)
    standard_time = time.time() - start_time
    
    results.append({
        "mode": "Standard",
        "time": standard_time,
        "throughput": len(test_df) / standard_time,
        "memory": "High (full dataset in memory)",
        "parallelism": "Single-threaded"
    })
    
    # Streaming processing
    try:
        from cleanepi.performance.streaming import StreamingCleaner
        
        cleaner = StreamingCleaner(chunk_size=2000)
        start_time = time.time()
        output_path = temp_dir / "comparison_streaming.csv"
        _, _ = cleaner.clean_csv_streaming(test_file, config, output_path=output_path)
        streaming_time = time.time() - start_time
        
        results.append({
            "mode": "Streaming",
            "time": streaming_time,
            "throughput": len(test_df) / streaming_time,
            "memory": "Low (chunk-based)",
            "parallelism": "Single-threaded"
        })
    except ImportError:
        pass
    
    # Async processing
    try:
        from cleanepi.performance.async_processing import AsyncCleaner
        
        async def run_async_test():
            async with AsyncCleaner(max_workers=2) as async_cleaner:
                output_path = temp_dir / "comparison_async.csv"
                _, _ = await async_cleaner.clean_csv(test_file, config, output_path=output_path)
        
        start_time = time.time()
        asyncio.run(run_async_test())
        async_time = time.time() - start_time
        
        results.append({
            "mode": "Async",
            "time": async_time,
            "throughput": len(test_df) / async_time,
            "memory": "Medium",
            "parallelism": "Multi-threaded"
        })
    except ImportError:
        pass
    
    # Display comparison
    print("\nProcessing Mode Comparison:")
    print(f"{'Mode':<12} {'Time (s)':<10} {'Throughput':<15} {'Memory':<20} {'Parallelism'}")
    print("-" * 75)
    
    for result in results:
        throughput_str = f"{result['throughput']:,.0f} rows/s"
        print(f"{result['mode']:<12} {result['time']:<10.3f} {throughput_str:<15} {result['memory']:<20} {result['parallelism']}")


def main():
    """Run all Phase 3 demonstrations."""
    try:
        print("Starting Phase 3 performance demonstrations...\n")
        
        # Run demonstrations
        demonstrate_standard_processing()
        demonstrate_streaming_processing()
        
        # Run async demo
        asyncio.run(demonstrate_async_processing())
        
        demonstrate_dask_processing()
        demonstrate_benchmarking()
        demonstrate_processing_modes_comparison()
        
        print("\n\n" + "=" * 60)
        print("PHASE 3 DEMONSTRATION COMPLETE")
        print("=" * 60)
        print()
        print("Key Phase 3 Features Demonstrated:")
        print("✓ Streaming processing for memory-efficient large dataset handling")
        print("✓ Asynchronous processing for non-blocking operations")
        print("✓ Dask integration for distributed computing")
        print("✓ Comprehensive performance benchmarking")
        print("✓ Multiple processing mode comparisons")
        print("✓ Scalability testing across different data sizes")
        print()
        print("These features enable cleanepi to scale from small research")
        print("datasets to large population-level epidemiological data while")
        print("maintaining high performance and reliability.")
        print()
        print(f"Temporary files and results saved in: {temp_dir}")
        print("You can examine the output files and benchmark results.")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up option
        cleanup = input("\nClean up temporary files? (y/N): ").lower().strip()
        if cleanup == 'y':
            shutil.rmtree(temp_dir)
            print("Temporary files cleaned up.")
        else:
            print(f"Temporary files preserved in: {temp_dir}")


if __name__ == "__main__":
    main()