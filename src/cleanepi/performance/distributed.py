"""
Distributed processing capabilities for cleanepi.

This module provides distributed processing using Dask.distributed
for scaling across multiple machines and handling fault tolerance.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import time
import uuid

try:
    from dask.distributed import Client, Future, as_completed, wait
    from dask.distributed import get_worker, get_client
    import dask
    DASK_DISTRIBUTED_AVAILABLE = True
except ImportError:
    DASK_DISTRIBUTED_AVAILABLE = False
    Client = None
    Future = None

import pandas as pd
from loguru import logger

from ..core.config import CleaningConfig
from ..core.report import CleaningReport
from ..core.clean_data import clean_data


class DistributedCleaner:
    """
    Distributed data cleaner using Dask.distributed.
    
    This class provides fault-tolerant distributed processing capabilities
    for large-scale data cleaning operations across multiple machines.
    """
    
    def __init__(
        self,
        scheduler_address: Optional[str] = None,
        client: Optional[Client] = None,
        timeout: int = 300
    ):
        """
        Initialize DistributedCleaner.
        
        Parameters
        ----------
        scheduler_address : str, optional
            Address of Dask scheduler. If None, creates local cluster.
        client : dask.distributed.Client, optional
            Existing Dask client. If provided, scheduler_address is ignored.
        timeout : int, default 300
            Timeout in seconds for distributed operations.
        """
        if not DASK_DISTRIBUTED_AVAILABLE:
            raise ImportError(
                "Distributed processing requires 'dask.distributed' to be installed. "
                "Install with: pip install 'cleanepi-python[performance]'"
            )
        
        self.timeout = timeout
        self.task_id = str(uuid.uuid4())[:8]
        
        if client is not None:
            self.client = client
            self.owns_client = False
        elif scheduler_address is not None:
            self.client = Client(scheduler_address, timeout=timeout)
            self.owns_client = True
        else:
            # Create local cluster
            from dask.distributed import LocalCluster
            cluster = LocalCluster(n_workers=2, threads_per_worker=2)
            self.client = Client(cluster)
            self.owns_client = True
        
        logger.info(f"DistributedCleaner initialized with scheduler: {self.client.scheduler_info()['address']}")
    
    def clean_dataframe_distributed(
        self,
        df: pd.DataFrame,
        config: CleaningConfig,
        partition_size: int = 10000,
        progress_callback: Optional[callable] = None
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        """
        Clean a DataFrame using distributed processing.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to clean.
        config : CleaningConfig
            Configuration for cleaning operations.
        partition_size : int, default 10000
            Number of rows per partition for distributed processing.
        progress_callback : callable, optional
            Callback function to report progress.
            
        Returns
        -------
        tuple
            Cleaned DataFrame and aggregated cleaning report.
        """
        logger.info(f"Starting distributed cleaning of DataFrame with {len(df)} rows")
        
        start_time = time.time()
        
        # Partition the DataFrame
        partitions = []
        total_partitions = (len(df) + partition_size - 1) // partition_size
        
        for i in range(0, len(df), partition_size):
            partition = df.iloc[i:i + partition_size].copy()
            partitions.append(partition)
        
        logger.info(f"Created {len(partitions)} partitions for distributed processing")
        
        if progress_callback:
            progress_callback(0, len(partitions), "Submitting tasks to workers")
        
        # Submit tasks to workers
        futures = []
        for i, partition in enumerate(partitions):
            future = self.client.submit(
                self._clean_partition,
                partition,
                config,
                task_id=f"{self.task_id}-{i}",
                retries=2,
                key=f"clean-partition-{self.task_id}-{i}"
            )
            futures.append(future)
        
        # Wait for completion and collect results
        cleaned_partitions = []
        reports = []
        completed = 0
        
        for future in as_completed(futures, timeout=self.timeout):
            try:
                cleaned_partition, report = future.result()
                cleaned_partitions.append(cleaned_partition)
                reports.append(report)
                completed += 1
                
                if progress_callback:
                    progress_callback(
                        completed,
                        len(partitions),
                        f"Completed partition {completed}/{len(partitions)}"
                    )
                
            except Exception as e:
                logger.error(f"Error processing partition: {str(e)}")
                completed += 1
                
                # Create error report for failed partition
                error_report = CleaningReport(initial_shape=(0, 0))
                error_report.metadata["error"] = str(e)
                reports.append(error_report)
        
        # Combine results
        if cleaned_partitions:
            final_df = pd.concat(cleaned_partitions, ignore_index=True)
        else:
            final_df = pd.DataFrame()
        
        # Create aggregated report
        aggregated_report = self._aggregate_reports(reports)
        aggregated_report.metadata["distributed_processing"] = True
        aggregated_report.metadata["total_partitions"] = len(partitions)
        aggregated_report.metadata["partition_size"] = partition_size
        aggregated_report.metadata["worker_count"] = len(self.client.scheduler_info()["workers"])
        aggregated_report.performance_metrics["distributed_processing_time"] = time.time() - start_time
        
        logger.info(f"Distributed cleaning completed: {len(df)} -> {len(final_df)} rows")
        
        return final_df, aggregated_report
    
    def clean_files_distributed(
        self,
        file_paths: List[Union[str, Path]],
        config: CleaningConfig,
        output_dir: Optional[Union[str, Path]] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, CleaningReport]:
        """
        Clean multiple files using distributed processing.
        
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
            
        Returns
        -------
        dict
            Dictionary mapping file paths to their cleaning reports.
        """
        logger.info(f"Starting distributed file processing: {len(file_paths)} files")
        
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        if progress_callback:
            progress_callback(0, len(file_paths), "Submitting file processing tasks")
        
        # Submit file processing tasks
        futures = {}
        for i, file_path in enumerate(file_paths):
            output_path = None
            if output_dir is not None:
                output_path = output_dir / f"cleaned_{Path(file_path).name}"
            
            future = self.client.submit(
                self._clean_file,
                file_path,
                config,
                output_path,
                task_id=f"{self.task_id}-file-{i}",
                retries=1,
                key=f"clean-file-{self.task_id}-{i}"
            )
            futures[future] = str(file_path)
        
        # Collect results
        results = {}
        completed = 0
        
        for future in as_completed(futures.keys(), timeout=self.timeout):
            file_path = futures[future]
            
            try:
                report = future.result()
                results[file_path] = report
                completed += 1
                
                if progress_callback:
                    progress_callback(
                        completed,
                        len(file_paths),
                        f"Completed file {completed}/{len(file_paths)}: {Path(file_path).name}"
                    )
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                
                # Create error report
                error_report = CleaningReport(initial_shape=(0, 0))
                error_report.metadata["error"] = str(e)
                error_report.metadata["file_path"] = file_path
                results[file_path] = error_report
                completed += 1
        
        logger.info(f"Distributed file processing completed: {len(file_paths)} files")
        
        return results
    
    def process_large_csv_distributed(
        self,
        file_path: Union[str, Path],
        config: CleaningConfig,
        output_path: Optional[Union[str, Path]] = None,
        chunk_size: int = 50000,
        progress_callback: Optional[callable] = None
    ) -> Tuple[Optional[pd.DataFrame], CleaningReport]:
        """
        Process a large CSV file using distributed chunking.
        
        Parameters
        ----------
        file_path : str or Path
            Path to input CSV file.
        config : CleaningConfig
            Configuration for cleaning operations.
        output_path : str or Path, optional
            Path to save cleaned data.
        chunk_size : int, default 50000
            Number of rows per chunk.
        progress_callback : callable, optional
            Callback function to report progress.
            
        Returns
        -------
        tuple
            Cleaned DataFrame (if output_path is None) and cleaning report.
        """
        logger.info(f"Starting distributed CSV processing: {file_path}")
        
        # First, split the file into chunks and upload to workers
        chunk_futures = []
        chunk_count = 0
        
        if progress_callback:
            progress_callback(0, -1, "Reading and distributing CSV chunks")
        
        try:
            # Read CSV in chunks and submit each chunk for processing
            for chunk_df in pd.read_csv(file_path, chunksize=chunk_size):
                future = self.client.submit(
                    self._clean_partition,
                    chunk_df,
                    config,
                    task_id=f"{self.task_id}-chunk-{chunk_count}",
                    retries=2,
                    key=f"clean-csv-chunk-{self.task_id}-{chunk_count}"
                )
                chunk_futures.append(future)
                chunk_count += 1
        
        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
            raise
        
        logger.info(f"Submitted {chunk_count} chunks for distributed processing")
        
        # Collect results
        cleaned_chunks = []
        reports = []
        completed = 0
        
        for future in as_completed(chunk_futures, timeout=self.timeout):
            try:
                cleaned_chunk, report = future.result()
                cleaned_chunks.append(cleaned_chunk)
                reports.append(report)
                completed += 1
                
                if progress_callback:
                    progress_callback(
                        completed,
                        chunk_count,
                        f"Completed chunk {completed}/{chunk_count}"
                    )
                
            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
                completed += 1
        
        # Combine results
        if cleaned_chunks:
            final_df = pd.concat(cleaned_chunks, ignore_index=True)
        else:
            final_df = pd.DataFrame()
        
        # Save or return results
        if output_path is not None:
            final_df.to_csv(output_path, index=False)
            result_df = None
        else:
            result_df = final_df
        
        # Create aggregated report
        aggregated_report = self._aggregate_reports(reports)
        aggregated_report.metadata["distributed_csv_processing"] = True
        aggregated_report.metadata["total_chunks"] = chunk_count
        aggregated_report.metadata["chunk_size"] = chunk_size
        
        logger.info(f"Distributed CSV processing completed: {chunk_count} chunks")
        
        return result_df, aggregated_report
    
    @staticmethod
    def _clean_partition(
        partition: pd.DataFrame,
        config: CleaningConfig,
        task_id: Optional[str] = None
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        """
        Clean a single partition (executed on worker).
        
        This method runs on distributed workers and should be static.
        """
        try:
            worker = get_worker()
            logger.info(f"Processing partition on worker {worker.address} (task: {task_id})")
        except:
            logger.info(f"Processing partition (task: {task_id})")
        
        start_time = time.time()
        
        try:
            cleaned_df, report = clean_data(partition, config)
            
            # Add worker information to report
            report.performance_metrics["worker_processing_time"] = time.time() - start_time
            report.performance_metrics["partition_rows"] = len(partition)
            report.performance_metrics["task_id"] = task_id
            
            try:
                worker = get_worker()
                report.performance_metrics["worker_address"] = worker.address
            except:
                pass
            
            return cleaned_df, report
            
        except Exception as e:
            logger.error(f"Error in partition processing (task: {task_id}): {str(e)}")
            
            # Return original data with error report
            error_report = CleaningReport(initial_shape=(0, 0))
            error_report.metadata["error"] = str(e)
            error_report.metadata["task_id"] = task_id
            
            return partition, error_report
    
    @staticmethod
    def _clean_file(
        file_path: Union[str, Path],
        config: CleaningConfig,
        output_path: Optional[Union[str, Path]] = None,
        task_id: Optional[str] = None
    ) -> CleaningReport:
        """
        Clean a single file (executed on worker).
        
        This method runs on distributed workers and should be static.
        """
        try:
            worker = get_worker()
            logger.info(f"Processing file {file_path} on worker {worker.address} (task: {task_id})")
        except:
            logger.info(f"Processing file {file_path} (task: {task_id})")
        
        start_time = time.time()
        
        try:
            # Read file
            if str(file_path).endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)
            
            # Clean data
            cleaned_df, report = clean_data(df, config)
            
            # Save if output path provided
            if output_path is not None:
                if str(output_path).endswith('.parquet'):
                    cleaned_df.to_parquet(output_path, index=False)
                else:
                    cleaned_df.to_csv(output_path, index=False)
            
            # Add worker information to report
            report.performance_metrics["worker_processing_time"] = time.time() - start_time
            report.performance_metrics["file_rows"] = len(df)
            report.performance_metrics["task_id"] = task_id
            report.performance_metrics["input_file"] = str(file_path)
            report.performance_metrics["output_file"] = str(output_path) if output_path else None
            
            try:
                worker = get_worker()
                report.performance_metrics["worker_address"] = worker.address
            except:
                pass
            
            return report
            
        except Exception as e:
            logger.error(f"Error processing file {file_path} (task: {task_id}): {str(e)}")
            
            # Create error report
            error_report = CleaningReport(initial_shape=(0, 0))
            error_report.metadata["error"] = str(e)
            error_report.metadata["file_path"] = str(file_path)
            error_report.metadata["task_id"] = task_id
            
            return error_report
    
    def _aggregate_reports(self, reports: List[CleaningReport]) -> CleaningReport:
        """Aggregate multiple cleaning reports."""
        if not reports:
            return CleaningReport(initial_shape=(0, 0))
        
        # Use the initial shape from the first report that's not an error
        first_valid_report = next((r for r in reports if "error" not in r.metadata), reports[0])
        aggregated = CleaningReport(initial_shape=first_valid_report.initial_shape)
        
        # Aggregate summary statistics
        total_rows = sum(r.metadata.get("total_rows", 0) for r in reports)
        error_count = sum(1 for r in reports if "error" in r.metadata)
        
        aggregated.metadata["total_rows"] = total_rows
        aggregated.metadata["total_partitions"] = len(reports)
        aggregated.metadata["error_count"] = error_count
        aggregated.metadata["success_count"] = len(reports) - error_count
        
        # Aggregate performance metrics
        processing_times = [
            r.performance_metrics.get("worker_processing_time", 0) for r in reports
            if "error" not in r.metadata
        ]
        
        if processing_times:
            aggregated.performance_metrics["total_worker_time"] = sum(processing_times)
            aggregated.performance_metrics["average_worker_time"] = sum(processing_times) / len(processing_times)
            aggregated.performance_metrics["max_worker_time"] = max(processing_times)
            aggregated.performance_metrics["min_worker_time"] = min(processing_times)
        
        # Collect worker addresses
        worker_addresses = set()
        for report in reports:
            if "worker_address" in report.performance_metrics:
                worker_addresses.add(report.performance_metrics["worker_address"])
        
        aggregated.performance_metrics["workers_used"] = list(worker_addresses)
        aggregated.performance_metrics["worker_count"] = len(worker_addresses)
        
        return aggregated
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """
        Get current cluster status and performance metrics.
        
        Returns
        -------
        dict
            Dictionary containing cluster status information.
        """
        try:
            scheduler_info = self.client.scheduler_info()
            workers = scheduler_info.get("workers", {})
            
            # Calculate total resources
            total_cores = sum(worker.get("nthreads", 0) for worker in workers.values())
            total_memory = sum(worker.get("memory_limit", 0) for worker in workers.values())
            used_memory = sum(worker.get("metrics", {}).get("memory", 0) for worker in workers.values())
            
            # Get task information
            tasks = self.client.scheduler_info().get("tasks", {})
            
            return {
                "scheduler_address": scheduler_info.get("address"),
                "num_workers": len(workers),
                "total_cores": total_cores,
                "total_memory_gb": total_memory / (1024**3) if total_memory else 0,
                "used_memory_gb": used_memory / (1024**3) if used_memory else 0,
                "memory_utilization": (used_memory / total_memory * 100) if total_memory else 0,
                "active_tasks": len([t for t in tasks.values() if t.get("state") == "executing"]),
                "pending_tasks": len([t for t in tasks.values() if t.get("state") == "waiting"]),
                "completed_tasks": len([t for t in tasks.values() if t.get("state") == "finished"]),
                "failed_tasks": len([t for t in tasks.values() if t.get("state") == "error"]),
                "client_status": str(self.client.status),
            }
        except Exception as e:
            logger.error(f"Error getting cluster status: {str(e)}")
            return {"error": str(e)}
    
    def close(self):
        """Close the distributed client."""
        if self.client is not None and self.owns_client:
            logger.info("Closing distributed client")
            self.client.close()
            self.client = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()