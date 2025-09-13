"""
Async processing capabilities for cleanepi.

This module provides asynchronous implementations of cleanepi functions
for non-blocking data processing in web applications and concurrent workflows.
"""

import asyncio
import io
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import concurrent.futures
import time

try:
    import aiofiles
    import aiofiles.os
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

import pandas as pd
from loguru import logger

from ..core.config import CleaningConfig
from ..core.report import CleaningReport
from ..core.clean_data import clean_data


class AsyncCleaner:
    """
    Async data cleaner for non-blocking operations.
    
    This class provides asynchronous implementations of cleaning operations
    that can be used in web applications and other async contexts.
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize AsyncCleaner.
        
        Parameters
        ----------
        max_workers : int, default 4
            Maximum number of worker threads for CPU-bound operations.
        """
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        )
        logger.info(f"AsyncCleaner initialized with {max_workers} workers")
    
    async def clean_dataframe(
        self,
        df: pd.DataFrame,
        config: CleaningConfig,
        progress_callback: Optional[callable] = None
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        """
        Clean a DataFrame asynchronously.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to clean.
        config : CleaningConfig
            Configuration for cleaning operations.
        progress_callback : callable, optional
            Callback function to report progress. Should accept (step, total_steps, message).
            
        Returns
        -------
        tuple
            Cleaned DataFrame and cleaning report.
        """
        start_time = time.time()
        logger.info(f"Starting async cleaning of DataFrame with {len(df)} rows")
        
        if progress_callback:
            await progress_callback(0, 1, "Starting data cleaning...")
        
        # Run the cleaning operation in the thread pool
        loop = asyncio.get_event_loop()
        cleaned_df, report = await loop.run_in_executor(
            self.executor,
            clean_data,
            df,
            config
        )
        
        # Add async processing metadata to report
        end_time = time.time()
        report.performance_metrics["async_processing_time"] = end_time - start_time
        report.performance_metrics["worker_threads"] = self.max_workers
        
        if progress_callback:
            await progress_callback(1, 1, "Data cleaning completed")
        
        logger.info(f"Async cleaning completed in {end_time - start_time:.2f} seconds")
        
        return cleaned_df, report
    
    async def clean_csv(
        self,
        file_path: Union[str, Path],
        config: CleaningConfig,
        output_path: Optional[Union[str, Path]] = None,
        progress_callback: Optional[callable] = None,
        **read_csv_kwargs
    ) -> Tuple[Optional[pd.DataFrame], CleaningReport]:
        """
        Clean a CSV file asynchronously.
        
        Parameters
        ----------
        file_path : str or Path
            Path to input CSV file.
        config : CleaningConfig
            Configuration for cleaning operations.
        output_path : str or Path, optional
            Path to save cleaned data. If None, returns DataFrame in memory.
        progress_callback : callable, optional
            Callback function to report progress.
        **read_csv_kwargs
            Additional arguments passed to pandas.read_csv.
            
        Returns
        -------
        tuple
            Cleaned DataFrame (if output_path is None) and cleaning report.
        """
        start_time = time.time()
        logger.info(f"Starting async CSV processing: {file_path}")
        
        if progress_callback:
            await progress_callback(0, 3, f"Reading CSV file: {file_path}")
        
        # Read CSV asynchronously
        loop = asyncio.get_event_loop()
        
        try:
            df = await loop.run_in_executor(
                self.executor,
                lambda: pd.read_csv(file_path, **read_csv_kwargs)
            )
        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
            raise
        
        if progress_callback:
            await progress_callback(1, 3, f"Cleaning data ({len(df)} rows)")
        
        # Clean the DataFrame
        cleaned_df, report = await self.clean_dataframe(
            df, config, progress_callback=None  # Don't double-report progress
        )
        
        if progress_callback:
            await progress_callback(2, 3, "Saving results")
        
        # Save results if output path provided
        if output_path is not None:
            await self._save_dataframe_async(cleaned_df, output_path)
            
            # Add file operation metadata
            report.performance_metrics["output_file"] = str(output_path)
            
            if progress_callback:
                await progress_callback(3, 3, f"Results saved to: {output_path}")
            
            return None, report
        else:
            if progress_callback:
                await progress_callback(3, 3, "Processing completed")
            
            return cleaned_df, report
    
    async def clean_multiple_files(
        self,
        file_paths: List[Union[str, Path]],
        config: CleaningConfig,
        output_dir: Optional[Union[str, Path]] = None,
        progress_callback: Optional[callable] = None,
        max_concurrent: int = 3
    ) -> List[Tuple[str, CleaningReport]]:
        """
        Clean multiple files concurrently.
        
        Parameters
        ----------
        file_paths : list
            List of file paths to process.
        config : CleaningConfig
            Configuration for cleaning operations.
        output_dir : str or Path, optional
            Directory to save cleaned files. If None, files are processed in memory.
        progress_callback : callable, optional
            Callback function to report progress.
        max_concurrent : int, default 3
            Maximum number of files to process concurrently.
            
        Returns
        -------
        list
            List of tuples containing (file_path, cleaning_report) for each file.
        """
        logger.info(f"Starting concurrent processing of {len(file_paths)} files")
        
        if progress_callback:
            await progress_callback(0, len(file_paths), "Starting batch processing")
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_file(file_path: Union[str, Path]) -> Tuple[str, CleaningReport]:
            async with semaphore:
                try:
                    # Determine output path
                    if output_dir is not None:
                        output_path = Path(output_dir) / f"cleaned_{Path(file_path).name}"
                    else:
                        output_path = None
                    
                    # Process the file
                    _, report = await self.clean_csv(
                        file_path,
                        config,
                        output_path=output_path
                    )
                    
                    return str(file_path), report
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    
                    # Create error report
                    error_report = CleaningReport(initial_shape=(0, 0))
                    error_report.metadata["error"] = str(e)
                    error_report.metadata["file_path"] = str(file_path)
                    
                    return str(file_path), error_report
        
        # Process all files concurrently
        tasks = [process_single_file(file_path) for file_path in file_paths]
        
        results = []
        completed = 0
        
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            
            if progress_callback:
                await progress_callback(
                    completed,
                    len(file_paths),
                    f"Completed {completed}/{len(file_paths)} files"
                )
        
        logger.info(f"Batch processing completed: {len(results)} files processed")
        
        return results
    
    async def clean_streaming(
        self,
        data_stream: asyncio.StreamReader,
        config: CleaningConfig,
        chunk_size: int = 10000,
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[pd.DataFrame], CleaningReport]:
        """
        Clean data from a streaming source.
        
        Parameters
        ----------
        data_stream : asyncio.StreamReader
            Stream of data to clean.
        config : CleaningConfig
            Configuration for cleaning operations.
        chunk_size : int, default 10000
            Number of rows to process in each chunk.
        progress_callback : callable, optional
            Callback function to report progress.
            
        Returns
        -------
        tuple
            List of cleaned DataFrame chunks and aggregated cleaning report.
        """
        logger.info("Starting streaming data processing")
        
        chunks = []
        reports = []
        chunk_count = 0
        
        if progress_callback:
            await progress_callback(0, -1, "Starting streaming processing")
        
        try:
            while True:
                # Read chunk from stream (simplified - would need actual CSV parsing)
                chunk_data = await data_stream.read(chunk_size)
                
                if not chunk_data:
                    break
                
                # Convert chunk to DataFrame (this is simplified)
                # In practice, you'd need proper CSV parsing logic
                try:
                    # This is a placeholder - real implementation would parse CSV data
                    df_chunk = pd.read_csv(io.StringIO(chunk_data.decode()))
                    
                    # Clean the chunk
                    cleaned_chunk, report = await self.clean_dataframe(df_chunk, config)
                    
                    chunks.append(cleaned_chunk)
                    reports.append(report)
                    chunk_count += 1
                    
                    if progress_callback:
                        await progress_callback(
                            chunk_count,
                            -1,
                            f"Processed {chunk_count} chunks"
                        )
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_count}: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Error in streaming processing: {str(e)}")
            raise
        
        # Aggregate reports
        aggregated_report = self._aggregate_reports(reports)
        aggregated_report.metadata["streaming_chunks"] = chunk_count
        
        if progress_callback:
            await progress_callback(chunk_count, chunk_count, "Streaming processing completed")
        
        logger.info(f"Streaming processing completed: {chunk_count} chunks processed")
        
        return chunks, aggregated_report
    
    async def _save_dataframe_async(
        self,
        df: pd.DataFrame,
        output_path: Union[str, Path]
    ):
        """Save DataFrame asynchronously."""
        if not AIOFILES_AVAILABLE:
            # Fallback to synchronous save in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                lambda: df.to_csv(output_path, index=False)
            )
        else:
            # Use aiofiles for truly async file operations
            if str(output_path).endswith('.csv'):
                csv_data = df.to_csv(index=False)
                async with aiofiles.open(output_path, 'w') as f:
                    await f.write(csv_data)
            else:
                # For other formats, fall back to thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.executor,
                    lambda: df.to_csv(output_path, index=False)
                )
    
    def _aggregate_reports(self, reports: List[CleaningReport]) -> CleaningReport:
        """Aggregate multiple cleaning reports."""
        if not reports:
            return CleaningReport(initial_shape=(0, 0))
        
        # Use the initial shape from the first report
        first_report = reports[0]
        aggregated = CleaningReport(initial_shape=first_report.initial_shape)
        
        # Sum up numeric metrics
        total_rows = sum(r.metadata.get("total_rows", 0) for r in reports)
        total_operations = sum(len(r.operations) for r in reports)
        
        aggregated.metadata["total_rows"] = total_rows
        aggregated.metadata["total_operations"] = total_operations
        aggregated.metadata["chunks_processed"] = len(reports)
        
        # Aggregate performance metrics
        total_time = sum(
            r.performance_metrics.get("total_time", 0) for r in reports
        )
        aggregated.performance_metrics["total_time"] = total_time
        aggregated.performance_metrics["average_time_per_chunk"] = (
            total_time / len(reports) if reports else 0
        )
        
        return aggregated
    
    async def close(self):
        """Close the async cleaner and shutdown executor."""
        logger.info("Shutting down AsyncCleaner")
        self.executor.shutdown(wait=True)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()