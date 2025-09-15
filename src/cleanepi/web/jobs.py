"""
Job management system for async data cleaning operations.

This module provides job queuing, status tracking, and result storage
for concurrent data cleaning tasks.
"""

import uuid
import asyncio
import time
from typing import Dict, List, Optional, Any, Union
import tempfile
import os
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import pandas as pd
from loguru import logger

from ..core.clean_data import clean_data
from ..core.config import CleaningConfig


class JobStatus(str, Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"  
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobResult:
    """Container for job execution results."""
    job_id: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    original_filename: str = ""
    original_shape: Optional[tuple] = None
    cleaned_shape: Optional[tuple] = None
    config: Optional[Dict[str, Any]] = None
    report_summary: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    preview_data: Optional[List[Dict]] = None
    column_info: Optional[Dict[str, Any]] = None
    # path to the originally uploaded file persisted for this job (used for download/re-run)
    source_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert datetime objects to ISO format strings
        for field in ["created_at", "started_at", "completed_at"]:
            if result[field]:
                result[field] = result[field].isoformat()
        
        # Handle NaN/None values in preview_data and column_info
        if result.get("preview_data"):
            import pandas as pd
            import numpy as np
            
            # Replace NaN values with None for JSON serialization
            for row in result["preview_data"]:
                for key, value in row.items():
                    if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
                        row[key] = None
        
        if result.get("column_info") and result["column_info"].get("missing_values"):
            import pandas as pd
            import numpy as np
            
            # Replace NaN values with None
            for key, value in result["column_info"]["missing_values"].items():
                if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
                    result["column_info"]["missing_values"][key] = None
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobResult":
        """Create JobResult from dictionary."""
        # Convert ISO format strings back to datetime objects
        for field in ["created_at", "started_at", "completed_at"]:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])
        return cls(**data)


class JobManager:
    """Manages async data cleaning jobs."""
    
    def __init__(self, max_concurrent_jobs: int = 3, result_ttl_hours: int = 24):
        """
        Initialize job manager.
        
        Parameters
        ----------
        max_concurrent_jobs : int
            Maximum number of concurrent jobs
        result_ttl_hours : int
            Hours to keep job results before cleanup
        """
        self.max_concurrent_jobs = max_concurrent_jobs
        self.result_ttl = timedelta(hours=result_ttl_hours)
        self.jobs: Dict[str, JobResult] = {}
        self.job_queue = asyncio.Queue()
        self.running_jobs: Dict[str, asyncio.Task] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._worker_tasks: List[asyncio.Task] = []
        
    async def start(self):
        """Start job processing workers."""
        logger.info(f"Starting job manager with {self.max_concurrent_jobs} workers")
        
        # Start worker tasks
        for i in range(self.max_concurrent_jobs):
            task = asyncio.create_task(self._worker())
            task.set_name(f"job_worker_{i}")
            self._worker_tasks.append(task)
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_jobs())
        self._cleanup_task.set_name("job_cleanup")
    
    async def stop(self):
        """Stop job processing workers."""
        logger.info("Stopping job manager")
        
        # Cancel all worker tasks
        for task in self._worker_tasks:
            task.cancel()
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Wait for tasks to finish
        tasks: List[asyncio.Task] = list(self._worker_tasks)
        if self._cleanup_task is not None:
            tasks.append(self._cleanup_task)
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Cancel any running jobs
        for job_id, task in self.running_jobs.items():
            task.cancel()
            if job_id in self.jobs:
                self.jobs[job_id].status = JobStatus.CANCELLED
    
    async def submit_job(
        self, 
        data: pd.DataFrame, 
        config: CleaningConfig,
        filename: str = "uploaded_file",
        original_file_bytes: Optional[bytes] = None,
        original_file_ext: Optional[str] = None,
        temp_dir: Optional[str] = None,
    ) -> str:
        """
        Submit a new cleaning job.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to clean
        config : CleaningConfig
            Cleaning configuration
        filename : str
            Original filename
            
        Returns
        -------
        str
            Job ID for tracking
        """
        job_id = str(uuid.uuid4())
        
        # Create job result record
        # Optionally persist original file to a temp path for later download/reprocessing
        source_path = None
        if original_file_bytes is not None and original_file_ext is not None:
            tmp_dir = temp_dir or tempfile.gettempdir()
            os.makedirs(tmp_dir, exist_ok=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=original_file_ext, dir=tmp_dir) as f:
                f.write(original_file_bytes)
                source_path = f.name

        job_result = JobResult(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            original_filename=filename,
            original_shape=data.shape,
            config=config.dict(),
            source_path=source_path,
        )
        
        self.jobs[job_id] = job_result
        
        # Add to queue
        await self.job_queue.put((job_id, data, config))
        
        logger.info(f"Submitted job {job_id} for file: {filename}")
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[JobResult]:
        """Get job status and results."""
        return self.jobs.get(job_id)
    
    async def list_jobs(
        self, 
        status_filter: Optional[JobStatus] = None,
        limit: int = 50
    ) -> List[JobResult]:
        """
        List jobs with optional filtering.
        
        Parameters
        ----------
        status_filter : JobStatus, optional
            Filter by job status
        limit : int
            Maximum number of jobs to return
            
        Returns
        -------
        List[JobResult]
            List of job results
        """
        jobs = list(self.jobs.values())
        
        # Filter by status if specified
        if status_filter:
            jobs = [job for job in jobs if job.status == status_filter]
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        return jobs[:limit]
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending or running job.
        
        Parameters
        ----------
        job_id : str
            Job ID to cancel
            
        Returns
        -------
        bool
            True if job was cancelled, False if not found or already completed
        """
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        
        # Can only cancel pending or running jobs
        if job.status not in [JobStatus.PENDING, JobStatus.RUNNING]:
            return False
        
        # Cancel running task if exists
        if job_id in self.running_jobs:
            self.running_jobs[job_id].cancel()
            del self.running_jobs[job_id]
        
        # Update status
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now()
        
        logger.info(f"Cancelled job {job_id}")
        return True
    
    async def _worker(self):
        """Worker coroutine to process jobs from queue."""
        while True:
            try:
                # Get job from queue
                job_id, data, config = await self.job_queue.get()
                
                if job_id not in self.jobs:
                    continue
                
                job = self.jobs[job_id]
                
                # Check if job was cancelled
                if job.status == JobStatus.CANCELLED:
                    self.job_queue.task_done()
                    continue
                
                # Start processing
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now()
                
                logger.info(f"Starting job {job_id}")
                
                # Create and track the processing task
                task = asyncio.create_task(self._process_job(job_id, data, config))
                self.running_jobs[job_id] = task
                
                try:
                    await task
                except asyncio.CancelledError:
                    job.status = JobStatus.CANCELLED
                    logger.info(f"Job {job_id} was cancelled")
                except Exception as e:
                    job.status = JobStatus.FAILED
                    job.error_message = str(e)
                    logger.error(f"Job {job_id} failed: {e}")
                finally:
                    # Clean up
                    if job_id in self.running_jobs:
                        del self.running_jobs[job_id]
                    job.completed_at = datetime.now()
                    if job.started_at:
                        job.processing_time = (job.completed_at - job.started_at).total_seconds()
                    
                    self.job_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    async def _process_job(self, job_id: str, data: pd.DataFrame, config: CleaningConfig):
        """Process a single cleaning job."""
        job = self.jobs[job_id]
        
        try:
            # Perform data cleaning
            cleaned_data, report = clean_data(data, config)
            
            # Store results
            job.status = JobStatus.COMPLETED
            job.cleaned_shape = cleaned_data.shape
            job.report_summary = report.summary()
            job.preview_data = cleaned_data.head(10).to_dict(orient="records")
            job.column_info = {
                "original_columns": list(data.columns),
                "cleaned_columns": list(cleaned_data.columns),
                "missing_values": cleaned_data.isna().sum().to_dict()
            }
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            logger.error(f"Job {job_id} processing failed: {e}")
            raise
    
    async def _cleanup_expired_jobs(self):
        """Periodically clean up expired job results."""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                current_time = datetime.now()
                expired_jobs = []
                
                for job_id, job in self.jobs.items():
                    if current_time - job.created_at > self.result_ttl:
                        expired_jobs.append(job_id)
                
                for job_id in expired_jobs:
                    job = self.jobs[job_id]
                    # Attempt to remove persisted source file if present
                    try:
                        if job.source_path and os.path.exists(job.source_path):
                            os.unlink(job.source_path)
                    except Exception:
                        pass
                    del self.jobs[job_id]
                    logger.info(f"Cleaned up expired job {job_id}")
                
                if expired_jobs:
                    logger.info(f"Cleaned up {len(expired_jobs)} expired jobs")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")


# Global job manager instance
_job_manager: Optional[JobManager] = None


async def get_job_manager() -> JobManager:
    """Get the global job manager instance."""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
        await _job_manager.start()
    return _job_manager


async def cleanup_job_manager():
    """Clean up the global job manager."""
    global _job_manager
    if _job_manager:
        await _job_manager.stop()
        _job_manager = None