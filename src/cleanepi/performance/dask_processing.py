"""
Dask integration for large-scale data processing.

This module provides Dask-based implementations of cleanepi functions
for processing datasets that don't fit in memory.
"""

from typing import Any, Dict, Optional, Tuple, Union
import logging
from pathlib import Path

try:
    import dask
    import dask.dataframe as dd
    from dask.dataframe import DataFrame as DaskDataFrame
    from dask.distributed import Client, as_completed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    DaskDataFrame = None

import pandas as pd
from loguru import logger

from ..core.config import CleaningConfig
from ..core.report import CleaningReport
from ..core.clean_data import clean_data


class DaskCleaner:
    """
    Dask-based data cleaner for large datasets.
    
    This class provides distributed and parallel processing capabilities
    for data cleaning operations that need to scale beyond single-machine
    memory limitations.
    """
    
    def __init__(self, client: Optional[Client] = None, memory_limit: str = "2GB"):
        """
        Initialize DaskCleaner.
        
        Parameters
        ----------
        client : dask.distributed.Client, optional
            Dask client for distributed processing. If None, a local client
            will be created.
        memory_limit : str, default "2GB"
            Memory limit per worker process.
        """
        if not DASK_AVAILABLE:
            raise ImportError(
                "Dask is required for DaskCleaner. "
                "Install with: pip install 'cleanepi-python[performance]'"
            )
        
        self.client = client
        self.memory_limit = memory_limit
        
        if self.client is None:
            # Create local client if none provided
            self.client = Client(
                n_workers=2,
                threads_per_worker=2,
                memory_limit=memory_limit,
                silence_logs=logging.WARNING
            )
            logger.info(f"Created local Dask client: {self.client}")
        
        logger.info(f"DaskCleaner initialized with client: {self.client}")
    
    def clean_dataframe(
        self,
        df: DaskDataFrame,
        config: CleaningConfig,
        chunk_size: Optional[int] = None
    ) -> Tuple[DaskDataFrame, CleaningReport]:
        """
        Clean a Dask DataFrame using distributed processing.
        
        Parameters
        ----------
        df : dask.dataframe.DataFrame
            Input Dask DataFrame to clean.
        config : CleaningConfig
            Configuration for cleaning operations.
        chunk_size : int, optional
            Size of chunks for processing. If None, uses Dask's default.
            
        Returns
        -------
        tuple
            Cleaned Dask DataFrame and aggregated cleaning report.
        """
        logger.info(f"Starting Dask cleaning of DataFrame with {df.npartitions} partitions")
        
        if chunk_size is not None:
            df = df.repartition(partition_size=f"{chunk_size}B")
        
        # Create a function that can be applied to each partition
        def clean_partition(partition_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
            """Clean a single partition and return results with metadata."""
            if partition_df.empty:
                return partition_df, {}
            
            try:
                cleaned_df, report = clean_data(partition_df, config)
                
                # Extract key metrics from report for aggregation
                metrics = {
                    "rows_processed": len(partition_df),
                    "rows_remaining": len(cleaned_df),
                    "columns_processed": len(partition_df.columns),
                    "columns_remaining": len(cleaned_df.columns),
                    "duplicates_removed": report.operations.get("remove_duplicates", {}).get("duplicates_removed", 0),
                    "constants_removed": len(report.operations.get("remove_constants", {}).get("removed_columns", [])),
                    "missing_values_replaced": report.operations.get("replace_missing_values", {}).get("values_replaced", 0),
                }
                
                return cleaned_df, metrics
            except Exception as e:
                logger.error(f"Error cleaning partition: {str(e)}")
                return partition_df, {"error": str(e)}
        
        # Apply cleaning to each partition
        logger.info("Applying cleaning operations to partitions...")
        
        # Use map_partitions for the cleaning operation
        cleaned_partitions = df.map_partitions(
            lambda partition: clean_partition(partition)[0],
            meta=df._meta
        )
        
        # Collect metadata from all partitions
        metadata_futures = []
        for i in range(df.npartitions):
            partition = df.get_partition(i)
            future = self.client.submit(
                lambda p: clean_partition(p.compute())[1],
                partition
            )
            metadata_futures.append(future)
        
        # Wait for all metadata and aggregate
        logger.info("Collecting metadata from all partitions...")
        aggregated_metrics = {
            "rows_processed": 0,
            "rows_remaining": 0,
            "columns_processed": 0,
            "columns_remaining": 0,
            "duplicates_removed": 0,
            "constants_removed": 0,
            "missing_values_replaced": 0,
            "errors": []
        }
        
        for future in as_completed(metadata_futures):
            try:
                metrics = future.result()
                if "error" in metrics:
                    aggregated_metrics["errors"].append(metrics["error"])
                else:
                    for key in ["rows_processed", "rows_remaining", "duplicates_removed", 
                              "constants_removed", "missing_values_replaced"]:
                        aggregated_metrics[key] += metrics.get(key, 0)
                    
                    # For columns, take the max (they should be the same across partitions)
                    aggregated_metrics["columns_processed"] = max(
                        aggregated_metrics["columns_processed"],
                        metrics.get("columns_processed", 0)
                    )
                    aggregated_metrics["columns_remaining"] = max(
                        aggregated_metrics["columns_remaining"],
                        metrics.get("columns_remaining", 0)
                    )
            except Exception as e:
                logger.error(f"Error collecting metadata: {str(e)}")
                aggregated_metrics["errors"].append(str(e))
        
        # Create aggregated report
        report = CleaningReport(initial_shape=(aggregated_metrics["rows_processed"], aggregated_metrics["columns_processed"]))
        report.metadata["total_rows_processed"] = aggregated_metrics["rows_processed"]
        report.metadata["total_rows_remaining"] = aggregated_metrics["rows_remaining"]
        report.metadata["total_duplicates_removed"] = aggregated_metrics["duplicates_removed"]
        report.metadata["total_constants_removed"] = aggregated_metrics["constants_removed"]
        report.metadata["total_missing_values_replaced"] = aggregated_metrics["missing_values_replaced"]
        report.metadata["processing_errors"] = aggregated_metrics["errors"]
        report.metadata["distributed_processing"] = True
        report.metadata["partitions_processed"] = df.npartitions
        
        logger.info(f"Dask cleaning completed. Processed {df.npartitions} partitions.")
        
        return cleaned_partitions, report
    
    def clean_csv(
        self,
        file_path: Union[str, Path],
        config: CleaningConfig,
        output_path: Optional[Union[str, Path]] = None,
        chunk_size: str = "100MB",
        **read_csv_kwargs
    ) -> Tuple[Optional[DaskDataFrame], CleaningReport]:
        """
        Clean a large CSV file using Dask.
        
        Parameters
        ----------
        file_path : str or Path
            Path to input CSV file.
        config : CleaningConfig
            Configuration for cleaning operations.
        output_path : str or Path, optional
            Path to save cleaned data. If None, returns DataFrame in memory.
        chunk_size : str, default "100MB"
            Size of chunks to read from CSV.
        **read_csv_kwargs
            Additional arguments passed to dask.dataframe.read_csv.
            
        Returns
        -------
        tuple
            Cleaned Dask DataFrame (if output_path is None) and cleaning report.
        """
        logger.info(f"Reading CSV file: {file_path}")
        
        # Read CSV with Dask
        df = dd.read_csv(
            file_path,
            blocksize=chunk_size,
            **read_csv_kwargs
        )
        
        logger.info(f"Loaded CSV with {df.npartitions} partitions")
        
        # Clean the DataFrame
        cleaned_df, report = self.clean_dataframe(df, config)
        
        # Save or return results
        if output_path is not None:
            logger.info(f"Saving cleaned data to: {output_path}")
            
            if str(output_path).endswith('.parquet'):
                cleaned_df.to_parquet(output_path)
            elif str(output_path).endswith('.csv'):
                cleaned_df.to_csv(output_path, index=False)
            else:
                # Default to parquet for better performance
                cleaned_df.to_parquet(f"{output_path}.parquet")
            
            return None, report
        else:
            return cleaned_df, report
    
    def clean_parquet(
        self,
        file_path: Union[str, Path],
        config: CleaningConfig,
        output_path: Optional[Union[str, Path]] = None
    ) -> Tuple[Optional[DaskDataFrame], CleaningReport]:
        """
        Clean a Parquet file using Dask.
        
        Parameters
        ----------
        file_path : str or Path
            Path to input Parquet file or directory.
        config : CleaningConfig
            Configuration for cleaning operations.
        output_path : str or Path, optional
            Path to save cleaned data. If None, returns DataFrame in memory.
            
        Returns
        -------
        tuple
            Cleaned Dask DataFrame (if output_path is None) and cleaning report.
        """
        logger.info(f"Reading Parquet file: {file_path}")
        
        # Read Parquet with Dask
        df = dd.read_parquet(file_path)
        
        logger.info(f"Loaded Parquet with {df.npartitions} partitions")
        
        # Clean the DataFrame
        cleaned_df, report = self.clean_dataframe(df, config)
        
        # Save or return results
        if output_path is not None:
            logger.info(f"Saving cleaned data to: {output_path}")
            cleaned_df.to_parquet(output_path)
            return None, report
        else:
            return cleaned_df, report
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics from Dask client.
        
        Returns
        -------
        dict
            Dictionary containing performance metrics.
        """
        if self.client is None:
            return {}
        
        try:
            # Get scheduler info
            scheduler_info = self.client.scheduler_info()
            
            # Get worker info
            workers = scheduler_info.get("workers", {})
            
            total_memory = sum(
                worker.get("memory_limit", 0) for worker in workers.values()
            )
            used_memory = sum(
                worker.get("memory", 0) for worker in workers.values()
            )
            
            total_cores = sum(
                worker.get("nthreads", 0) for worker in workers.values()
            )
            
            return {
                "num_workers": len(workers),
                "total_cores": total_cores,
                "total_memory_limit": total_memory,
                "used_memory": used_memory,
                "memory_utilization": used_memory / total_memory if total_memory > 0 else 0,
                "scheduler_address": scheduler_info.get("address"),
                "client_status": str(self.client.status),
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {"error": str(e)}
    
    def close(self):
        """Close the Dask client."""
        if self.client is not None:
            logger.info("Closing Dask client")
            self.client.close()
            self.client = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()