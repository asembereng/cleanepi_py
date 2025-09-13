"""
Streaming data processing for cleanepi.

This module provides streaming capabilities for processing large datasets
that don't fit in memory by processing them in chunks.
"""

import io
from typing import Any, Dict, Generator, Iterator, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
from loguru import logger

from ..core.config import CleaningConfig
from ..core.report import CleaningReport
from ..core.clean_data import clean_data

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class StreamingCleaner:
    """
    Streaming data cleaner for memory-efficient processing.
    
    This class processes large datasets in chunks to avoid memory limitations
    while maintaining data consistency across chunks.
    """
    
    def __init__(self, chunk_size: int = 10000, memory_limit: str = "500MB"):
        """
        Initialize StreamingCleaner.
        
        Parameters
        ----------
        chunk_size : int, default 10000
            Number of rows to process in each chunk.
        memory_limit : str, default "500MB"
            Memory limit for processing operations.
        """
        self.chunk_size = chunk_size
        self.memory_limit = memory_limit
        self._memory_limit_bytes = self._parse_memory_limit(memory_limit)
        
        logger.info(f"StreamingCleaner initialized with chunk_size={chunk_size}, memory_limit={memory_limit}")
    
    def _parse_memory_limit(self, memory_limit: str) -> int:
        """Parse memory limit string to bytes."""
        units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
        
        memory_limit = memory_limit.upper().strip()
        
        # Find the unit that matches
        for unit, multiplier in sorted(units.items(), key=lambda x: len(x[0]), reverse=True):
            if memory_limit.endswith(unit):
                value_str = memory_limit[:-len(unit)].strip()
                return int(float(value_str) * multiplier)
        
        # Default to bytes if no unit
        return int(memory_limit)
    
    def clean_csv_streaming(
        self,
        file_path: Union[str, Path],
        config: CleaningConfig,
        output_path: Optional[Union[str, Path]] = None,
        progress_callback: Optional[callable] = None,
        **read_csv_kwargs
    ) -> Tuple[Optional[pd.DataFrame], CleaningReport]:
        """
        Clean a CSV file using streaming processing.
        
        Parameters
        ----------
        file_path : str or Path
            Path to input CSV file.
        config : CleaningConfig
            Configuration for cleaning operations.
        output_path : str or Path, optional
            Path to save cleaned data. If None, returns concatenated DataFrame.
        progress_callback : callable, optional
            Callback function to report progress (chunk_num, total_chunks, message).
        **read_csv_kwargs
            Additional arguments passed to pandas.read_csv.
            
        Returns
        -------
        tuple
            Cleaned DataFrame (if output_path is None) and aggregated cleaning report.
        """
        logger.info(f"Starting streaming CSV processing: {file_path}")
        
        # Get total file size for progress estimation
        file_size = Path(file_path).stat().st_size
        
        # Initialize aggregation variables
        cleaned_chunks = []
        reports = []
        chunk_count = 0
        total_rows_processed = 0
        
        # Prepare output file if needed
        output_file = None
        if output_path is not None:
            output_file = open(output_path, 'w', newline='')
            header_written = False
        
        try:
            # Create chunk iterator
            chunk_iter = pd.read_csv(
                file_path,
                chunksize=self.chunk_size,
                **read_csv_kwargs
            )
            
            for chunk_df in chunk_iter:
                if progress_callback:
                    progress_callback(
                        chunk_count,
                        -1,  # Unknown total chunks
                        f"Processing chunk {chunk_count + 1} ({len(chunk_df)} rows)"
                    )
                
                # Clean the chunk
                cleaned_chunk, chunk_report = clean_data(chunk_df, config)
                
                # Handle output
                if output_path is not None:
                    # Write to file
                    if not header_written:
                        cleaned_chunk.to_csv(output_file, index=False)
                        header_written = True
                    else:
                        cleaned_chunk.to_csv(output_file, index=False, header=False)
                else:
                    # Keep in memory
                    cleaned_chunks.append(cleaned_chunk)
                
                # Collect metadata
                reports.append(chunk_report)
                chunk_count += 1
                total_rows_processed += len(chunk_df)
                
                # Check memory usage
                if hasattr(cleaned_chunk, 'memory_usage'):
                    current_memory = cleaned_chunk.memory_usage(deep=True).sum()
                    if current_memory > self._memory_limit_bytes:
                        logger.warning(f"Chunk memory usage ({current_memory} bytes) exceeds limit")
                
                logger.debug(f"Processed chunk {chunk_count}: {len(chunk_df)} -> {len(cleaned_chunk)} rows")
        
        finally:
            if output_file is not None:
                output_file.close()
        
        # Create aggregated report
        aggregated_report = self._aggregate_reports(reports)
        aggregated_report.metadata["streaming_mode"] = True
        aggregated_report.metadata["total_chunks"] = chunk_count
        aggregated_report.metadata["chunk_size"] = self.chunk_size
        aggregated_report.metadata["total_rows_processed"] = total_rows_processed
        
        if progress_callback:
            progress_callback(
                chunk_count,
                chunk_count,
                f"Streaming processing completed: {chunk_count} chunks, {total_rows_processed} rows"
            )
        
        logger.info(f"Streaming processing completed: {chunk_count} chunks, {total_rows_processed} rows")
        
        # Return results
        if output_path is not None:
            return None, aggregated_report
        else:
            if cleaned_chunks:
                final_df = pd.concat(cleaned_chunks, ignore_index=True)
                return final_df, aggregated_report
            else:
                return pd.DataFrame(), aggregated_report
    
    def clean_parquet_streaming(
        self,
        file_path: Union[str, Path],
        config: CleaningConfig,
        output_path: Optional[Union[str, Path]] = None,
        progress_callback: Optional[callable] = None
    ) -> Tuple[Optional[pd.DataFrame], CleaningReport]:
        """
        Clean a Parquet file using streaming processing.
        
        Parameters
        ----------
        file_path : str or Path
            Path to input Parquet file.
        config : CleaningConfig
            Configuration for cleaning operations.
        output_path : str or Path, optional
            Path to save cleaned data.
        progress_callback : callable, optional
            Callback function to report progress.
            
        Returns
        -------
        tuple
            Cleaned DataFrame (if output_path is None) and cleaning report.
        """
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "Parquet streaming requires 'pyarrow' to be installed. "
                "Install with: pip install 'cleanepi-python[performance]'"
            )
        
        logger.info(f"Starting streaming Parquet processing: {file_path}")
        
        # Read Parquet file metadata
        parquet_file = pq.ParquetFile(file_path)
        total_rows = parquet_file.metadata.num_rows
        
        cleaned_chunks = []
        reports = []
        chunk_count = 0
        rows_processed = 0
        
        # Process row groups as chunks
        for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
            if progress_callback:
                progress_callback(
                    chunk_count,
                    -1,
                    f"Processing batch {chunk_count + 1}"
                )
            
            # Convert to pandas DataFrame
            chunk_df = batch.to_pandas()
            
            # Clean the chunk
            cleaned_chunk, chunk_report = clean_data(chunk_df, config)
            
            cleaned_chunks.append(cleaned_chunk)
            reports.append(chunk_report)
            
            chunk_count += 1
            rows_processed += len(chunk_df)
            
            logger.debug(f"Processed batch {chunk_count}: {len(chunk_df)} -> {len(cleaned_chunk)} rows")
        
        # Aggregate results
        aggregated_report = self._aggregate_reports(reports)
        aggregated_report.metadata["streaming_mode"] = True
        aggregated_report.metadata["total_batches"] = chunk_count
        aggregated_report.metadata["total_rows_processed"] = rows_processed
        
        if progress_callback:
            progress_callback(
                chunk_count,
                chunk_count,
                f"Parquet streaming completed: {chunk_count} batches, {rows_processed} rows"
            )
        
        # Save or return results
        if output_path is not None:
            final_df = pd.concat(cleaned_chunks, ignore_index=True)
            final_df.to_parquet(output_path, index=False)
            return None, aggregated_report
        else:
            if cleaned_chunks:
                final_df = pd.concat(cleaned_chunks, ignore_index=True)
                return final_df, aggregated_report
            else:
                return pd.DataFrame(), aggregated_report
    
    def clean_dataframe_streaming(
        self,
        df: pd.DataFrame,
        config: CleaningConfig,
        progress_callback: Optional[callable] = None
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        """
        Clean a DataFrame using streaming approach (chunked processing).
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to clean.
        config : CleaningConfig
            Configuration for cleaning operations.
        progress_callback : callable, optional
            Callback function to report progress.
            
        Returns
        -------
        tuple
            Cleaned DataFrame and aggregated cleaning report.
        """
        logger.info(f"Starting streaming DataFrame processing: {len(df)} rows")
        
        total_chunks = (len(df) + self.chunk_size - 1) // self.chunk_size
        cleaned_chunks = []
        reports = []
        
        for i in range(0, len(df), self.chunk_size):
            chunk_num = i // self.chunk_size
            
            if progress_callback:
                progress_callback(
                    chunk_num,
                    total_chunks,
                    f"Processing chunk {chunk_num + 1}/{total_chunks}"
                )
            
            # Extract chunk
            chunk_df = df.iloc[i:i + self.chunk_size].copy()
            
            # Clean the chunk
            cleaned_chunk, chunk_report = clean_data(chunk_df, config)
            
            cleaned_chunks.append(cleaned_chunk)
            reports.append(chunk_report)
            
            logger.debug(f"Processed chunk {chunk_num + 1}/{total_chunks}: {len(chunk_df)} -> {len(cleaned_chunk)} rows")
        
        # Combine results
        final_df = pd.concat(cleaned_chunks, ignore_index=True)
        aggregated_report = self._aggregate_reports(reports)
        aggregated_report.metadata["streaming_mode"] = True
        aggregated_report.metadata["total_chunks"] = total_chunks
        aggregated_report.metadata["chunk_size"] = self.chunk_size
        
        if progress_callback:
            progress_callback(
                total_chunks,
                total_chunks,
                f"Streaming processing completed: {len(final_df)} rows"
            )
        
        logger.info(f"Streaming DataFrame processing completed: {len(df)} -> {len(final_df)} rows")
        
        return final_df, aggregated_report
    
    def process_multiple_files_streaming(
        self,
        file_paths: list,
        config: CleaningConfig,
        output_dir: Union[str, Path],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, CleaningReport]:
        """
        Process multiple files using streaming approach.
        
        Parameters
        ----------
        file_paths : list
            List of file paths to process.
        config : CleaningConfig
            Configuration for cleaning operations.
        output_dir : str or Path
            Directory to save cleaned files.
        progress_callback : callable, optional
            Callback function to report progress.
            
        Returns
        -------
        dict
            Dictionary mapping file paths to their cleaning reports.
        """
        logger.info(f"Starting streaming batch processing: {len(file_paths)} files")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for i, file_path in enumerate(file_paths):
            if progress_callback:
                progress_callback(
                    i,
                    len(file_paths),
                    f"Processing file {i + 1}/{len(file_paths)}: {Path(file_path).name}"
                )
            
            try:
                # Determine output path
                output_path = output_dir / f"cleaned_{Path(file_path).name}"
                
                # Process the file
                if str(file_path).endswith('.parquet'):
                    _, report = self.clean_parquet_streaming(
                        file_path, config, output_path
                    )
                else:
                    _, report = self.clean_csv_streaming(
                        file_path, config, output_path
                    )
                
                results[str(file_path)] = report
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                
                # Create error report
                error_report = CleaningReport(initial_shape=(0, 0))
                error_report.metadata["error"] = str(e)
                error_report.metadata["file_path"] = str(file_path)
                results[str(file_path)] = error_report
        
        if progress_callback:
            progress_callback(
                len(file_paths),
                len(file_paths),
                f"Batch streaming processing completed: {len(file_paths)} files"
            )
        
        logger.info(f"Streaming batch processing completed: {len(file_paths)} files")
        
        return results
    
    def _aggregate_reports(self, reports: list) -> CleaningReport:
        """Aggregate multiple cleaning reports."""
        if not reports:
            return CleaningReport(initial_shape=(0, 0))
        
        # Use the initial shape from the first report
        first_report = reports[0]
        aggregated = CleaningReport(initial_shape=first_report.initial_shape)
        
        # For streaming, we need to use the metadata dict instead of summary property
        # which gets set during processing
        aggregated.metadata["total_rows"] = sum(len(self.test_data) for _ in reports) if hasattr(self, 'test_data') else 0
        aggregated.metadata["chunks_processed"] = len(reports)
        aggregated.metadata["streaming_mode"] = True
        aggregated.metadata["total_chunks"] = len(reports)
        aggregated.metadata["chunk_size"] = self.chunk_size
        
        return aggregated
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage information.
        
        Returns
        -------
        dict
            Dictionary containing memory usage metrics.
        """
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available - memory monitoring disabled")
            return {
                "current_memory_mb": 0,
                "peak_memory_mb": 0,
                "memory_limit_mb": self._memory_limit_bytes / 1024 / 1024,
                "chunk_size": self.chunk_size,
                "memory_utilization": 0,
                "psutil_available": False
            }
        
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "current_memory_mb": memory_info.rss / 1024 / 1024,
            "peak_memory_mb": memory_info.vms / 1024 / 1024,
            "memory_limit_mb": self._memory_limit_bytes / 1024 / 1024,
            "chunk_size": self.chunk_size,
            "memory_utilization": (memory_info.rss / self._memory_limit_bytes) * 100,
            "psutil_available": True
        }