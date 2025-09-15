"""
FastAPI web application components for cleanepi.

This module provides REST API endpoints for data cleaning operations.
"""

from typing import Optional, Dict, Any, List
import tempfile
import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Depends
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import pandas as pd
from loguru import logger

from ..core.clean_data import clean_data
from ..core.config import (
    CleaningConfig,
    WebConfig,
    MissingValueConfig,
    DuplicateConfig,
    ConstantConfig,
)
from ..utils.validation import validate_file_safety, detect_encoding
from .jobs import JobManager, JobStatus, get_job_manager, cleanup_job_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan for job manager."""
    # Startup
    logger.info("Starting cleanepi web application")
    yield
    # Shutdown
    logger.info("Shutting down cleanepi web application")
    await cleanup_job_manager()


class CleaningAPI:
    """FastAPI application for data cleaning operations."""
    
    def __init__(self, config: Optional[WebConfig] = None):
        """Initialize the cleaning API."""
        self.config = config or WebConfig(
            max_file_size=100 * 1024 * 1024,
            allowed_file_types=[".csv", ".xlsx", ".parquet", ".json"],
            temp_dir="/tmp/cleanepi",
            enable_async=True,
            chunk_size=10000,
        )
        self.app = FastAPI(
            title="cleanepi API",
            description="Clean and standardize epidemiological data",
            version="0.1.0",
            lifespan=lifespan
        )
        
        # Setup templates and static files
        templates_dir = os.path.join(os.path.dirname(__file__), "templates")
        self.templates = Jinja2Templates(directory=templates_dir)
        # Ensure template changes are picked up without server restart in dev
        try:
            # Starlette's Jinja2Templates exposes env; enable auto-reload
            self.templates.env.auto_reload = True  # type: ignore[attr-defined]
            self.templates.env.cache = {}  # disable template cache in dev
        except Exception:
            pass
        
        self._setup_routes()
        
        # Mount static files (will be created later)
        try:
            static_dir = os.path.join(os.path.dirname(__file__), "static")
            self.app.mount("/static", StaticFiles(directory=static_dir), name="static")
        except Exception:
            logger.warning("Static files directory not found, skipping static file serving")

    @staticmethod
    def _default_cleaning_config() -> CleaningConfig:
        """Construct a default CleaningConfig with explicit defaults.
        This avoids static type checker complaints in some environments.
        """
        return CleaningConfig(
            standardize_column_names=True,
            replace_missing_values=MissingValueConfig(
                target_columns=None,
                na_strings=["-99", "N/A", "NULL", "", "missing", "unknown"],
                custom_na_by_column=None,
            ),
            remove_duplicates=DuplicateConfig(
                target_columns=None,
                subset=None,
                keep="first",
            ),
            remove_constants=ConstantConfig(
                cutoff=1.0,
                exclude_columns=None,
            ),
            standardize_dates=None,
            standardize_subject_ids=None,
            to_numeric=None,
            dictionary=None,
            check_date_sequence=None,
            verbose=True,
            strict_validation=False,
            max_memory_usage=None,
        )
    
    @staticmethod
    def _to_jsonable(obj: Any) -> Any:
        """Convert objects to JSON-serializable primitives.
        - Converts numpy types to native Python types
        - Converts pandas Timestamps/NaT to ISO strings/None
        - Replaces NaN/None-like with None
        - Recursively handles lists, tuples, sets, and dicts
        """
        try:
            import numpy as np  # type: ignore
        except Exception:  # pragma: no cover - numpy is a core dependency
            np = None  # type: ignore
        import pandas as pd  # already imported at module level
        from datetime import datetime, date

        # Fast path for common primitives
        if obj is None or isinstance(obj, (str, int, float, bool)):
            # Guard for float('nan')
            if isinstance(obj, float):
                try:
                    if obj != obj:  # NaN check
                        return None
                except Exception:
                    pass
            return obj

        # pandas NA/NaT or general missing
        try:
            if pd.isna(obj):  # type: ignore
                return None
        except Exception:
            pass

        # numpy scalars -> python
        if np is not None:
            if isinstance(obj, getattr(np, 'integer', ())):
                return int(obj)
            if isinstance(obj, getattr(np, 'floating', ())):
                try:
                    return None if np.isnan(obj) else float(obj)
                except Exception:
                    return float(obj)
            if isinstance(obj, getattr(np, 'bool_', ())):
                return bool(obj)

        # datetime-like
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        try:
            if isinstance(obj, pd.Timestamp):  # type: ignore
                return obj.isoformat()
        except Exception:
            pass

        # containers
        if isinstance(obj, dict):
            return {str(k): CleaningAPI._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [CleaningAPI._to_jsonable(v) for v in obj]

        # Fallback to string representation
        return str(obj)
    
    def _setup_routes(self):
        """Setup API routes."""
        
        # Web UI Routes
        @self.app.get("/", response_class=HTMLResponse)
        async def web_interface(request: Request):
            """Main web interface."""
            response = self.templates.TemplateResponse(
                "index.html",
                {"request": request, "title": "cleanepi - Data Cleaning Tool"}
            )
            # Prevent browser caching so template/JS changes show immediately
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            return response
        
        @self.app.get("/jobs", response_class=HTMLResponse)
        async def jobs_interface(request: Request):
            """Jobs management interface."""
            response = self.templates.TemplateResponse(
                "jobs.html", 
                {"request": request, "title": "Job Management"}
            )
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            return response
        
        @self.app.get("/config", response_class=HTMLResponse)
        async def config_interface(request: Request):
            """Configuration interface."""
            response = self.templates.TemplateResponse(
                "config.html",
                {"request": request, "title": "Configuration"}
            )
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            return response
        
        # API Routes
        @self.app.post("/api/clean")
        async def clean_data_endpoint(
            file: UploadFile = File(...),
            config_json: Optional[str] = Form(None)
        ):
            """
            Clean uploaded data file (synchronous).
            
            Parameters:
            - file: CSV or Excel file to clean
            - config_json: Optional JSON string with cleaning configuration
            """
            return await self._clean_data_handler(file, config_json)

        @self.app.post("/api/clean/download")
        async def clean_data_download(
            file: UploadFile = File(...),
            config_json: Optional[str] = Form(None)
        ):
            """
            Clean uploaded data file and return CSV as a download (synchronous).

            This reprocesses the uploaded file using the provided config and streams the
            cleaned result as a CSV attachment.
            """
            # Validate file and load data (reuse handler logic up to cleaning)
            # Create temporary file
            if not file:
                raise HTTPException(status_code=400, detail="No file provided")

            # Validate extension and size
            file_ext = os.path.splitext(file.filename or "")[1].lower()
            if file_ext not in self.config.allowed_file_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"File type not allowed. Allowed types: {self.config.allowed_file_types}"
                )

            # Note: UploadFile may not expose reliable size; enforce after reading content

            import io

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                content = await file.read()
                # Enforce max file size in bytes
                if self.config.max_file_size and len(content) > self.config.max_file_size:
                    tmp_file.close()
                    os.unlink(tmp_file.name)
                    raise HTTPException(status_code=413, detail="File too large")
                tmp_file.write(content)
                tmp_file_path = tmp_file.name

            try:
                # Validate file safety (allowed extensions and basic checks)
                validate_file_safety(tmp_file_path, self.config.allowed_file_types)

                # Load data
                if file_ext == '.csv':
                    encoding = detect_encoding(tmp_file_path)
                    data = pd.read_csv(tmp_file_path, encoding=encoding)
                elif file_ext in ['.xlsx', '.xls']:
                    data = pd.read_excel(tmp_file_path)
                elif file_ext == '.json':
                    data = pd.read_json(tmp_file_path)
                elif file_ext == '.parquet':
                    data = pd.read_parquet(tmp_file_path)
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")

                # Parse configuration
                if config_json:
                    config_dict = json.loads(config_json)
                    cleaning_config = CleaningConfig(**config_dict)
                else:
                    cleaning_config = self._default_cleaning_config()

                # Clean data
                cleaned_data, report = clean_data(data, cleaning_config)

                # Stream as CSV
                buffer = io.StringIO()
                cleaned_data.to_csv(buffer, index=False)
                buffer.seek(0)

                download_name = (file.filename or "cleanepi_output").rsplit('.', 1)[0] + "_cleaned.csv"
                headers = {"Content-Disposition": f"attachment; filename=\"{download_name}\""}
                return StreamingResponse(buffer, media_type="text/csv", headers=headers)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON configuration")
            except Exception as e:
                logger.error(f"Error generating download: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
            finally:
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
        
        @self.app.post("/api/jobs/submit")
        async def submit_cleaning_job(
            file: UploadFile = File(...),
            config_json: Optional[str] = Form(None),
            job_manager: JobManager = Depends(get_job_manager)
        ):
            """
            Submit async data cleaning job.
            
            Returns job ID for status checking.
            """
            return await self._submit_job_handler(file, config_json, job_manager)
        
        @self.app.get("/api/jobs/{job_id}")
        async def get_job_status(
            job_id: str,
            job_manager: JobManager = Depends(get_job_manager)
        ):
            """Get job status and results."""
            job_result = await job_manager.get_job_status(job_id)
            if not job_result:
                raise HTTPException(status_code=404, detail="Job not found")
            return job_result.to_dict()
        
        @self.app.get("/api/jobs")
        async def list_jobs(
            status: Optional[JobStatus] = None,
            limit: int = 50,
            job_manager: JobManager = Depends(get_job_manager)
        ):
            """List jobs with optional filtering."""
            jobs = await job_manager.list_jobs(status_filter=status, limit=limit)
            return [job.to_dict() for job in jobs]

        @self.app.get("/api/jobs/{job_id}/download")
        async def download_job_result(
            job_id: str,
            job_manager: JobManager = Depends(get_job_manager)
        ):
            """Download cleaned results for a completed job as CSV.

            Re-runs cleaning using the persisted source file and saved config to ensure a full dataset is returned.
            """
            job = await job_manager.get_job_status(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            if job.status != JobStatus.COMPLETED:
                raise HTTPException(status_code=400, detail="Job is not completed")
            if not job.source_path or not os.path.exists(job.source_path):
                raise HTTPException(status_code=410, detail="Original file unavailable for download")

            # Load original data from persisted file
            file_ext = os.path.splitext(job.source_path)[1].lower()
            try:
                if file_ext == '.csv':
                    encoding = detect_encoding(job.source_path)
                    data = pd.read_csv(job.source_path, encoding=encoding)
                elif file_ext in ['.xlsx', '.xls']:
                    data = pd.read_excel(job.source_path)
                elif file_ext == '.json':
                    data = pd.read_json(job.source_path)
                elif file_ext == '.parquet':
                    data = pd.read_parquet(job.source_path)
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")

                # Rebuild config
                config_dict = job.config or {}
                cleaning_config = CleaningConfig(**config_dict)

                # Clean data
                cleaned_data, _report = clean_data(data, cleaning_config)

                # Stream CSV
                import io
                buffer = io.StringIO()
                cleaned_data.to_csv(buffer, index=False)
                buffer.seek(0)
                base = (job.original_filename or 'cleanepi_output').rsplit('.', 1)[0]
                download_name = f"{base}_cleaned.csv"
                headers = {"Content-Disposition": f"attachment; filename=\"{download_name}\""}
                return StreamingResponse(buffer, media_type="text/csv", headers=headers)
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error preparing job download {job_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/api/jobs/{job_id}")
        async def cancel_job(
            job_id: str,
            job_manager: JobManager = Depends(get_job_manager)
        ):
            """Cancel a pending or running job."""
            success = await job_manager.cancel_job(job_id)
            if not success:
                raise HTTPException(
                    status_code=400, 
                    detail="Job not found or cannot be cancelled"
                )
            return {"message": f"Job {job_id} cancelled successfully"}
        
        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "version": "0.1.0"}
        
        @self.app.get("/api/config/default")
        async def get_default_config():
            """Get default cleaning configuration."""
            config = self._default_cleaning_config()
            return config.dict()
        
        # Legacy endpoints (for backward compatibility)
        @self.app.post("/clean")
        async def clean_data_endpoint_legacy(
            file: UploadFile = File(...),
            config_json: Optional[str] = None
        ):
            """Legacy clean endpoint for backward compatibility."""
            return await self._clean_data_handler(file, config_json)
        
        @self.app.post("/clean/async")
        async def clean_data_async_legacy(
            background_tasks: BackgroundTasks,
            file: UploadFile = File(...),
            config_json: Optional[str] = None,
            job_manager: JobManager = Depends(get_job_manager)
        ):
            """Legacy async endpoint - now redirects to job submission."""
            result = await self._submit_job_handler(file, config_json, job_manager)
            return {"job_id": result["job_id"], "message": "Job submitted successfully"}
        
        @self.app.get("/health")
        async def health_check_legacy():
            """Legacy health check endpoint."""
            return {"status": "healthy", "version": "0.1.0"}
        
        @self.app.get("/config/default")
        async def get_default_config_legacy():
            """Legacy config endpoint."""
            config = self._default_cleaning_config()
            return config.dict()
    
    async def _submit_job_handler(
        self,
        file: UploadFile,
        config_json: Optional[str],
        job_manager: JobManager
    ) -> Dict[str, Any]:
        """Handle async job submission."""
        # Basic checks
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")

        file_ext = os.path.splitext((file.filename or ""))[1].lower()
        if file_ext not in self.config.allowed_file_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {self.config.allowed_file_types}"
            )

        # Create temporary file and load data
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            # Enforce max file size (bytes)
            if self.config.max_file_size and len(content) > self.config.max_file_size:
                tmp_file.close()
                os.unlink(tmp_file.name)
                raise HTTPException(status_code=413, detail="File too large")
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # Validate file safety (size check in MB)
            try:
                max_mb = (self.config.max_file_size or (100 * 1024 * 1024)) / (1024 * 1024)
            except Exception:
                max_mb = 100
            validate_file_safety(tmp_file_path, self.config.allowed_file_types)

            # Load data
            if file_ext == '.csv':
                encoding = detect_encoding(tmp_file_path)
                data = pd.read_csv(tmp_file_path, encoding=encoding)
            elif file_ext in ['.xlsx', '.xls']:
                data = pd.read_excel(tmp_file_path)
            elif file_ext == '.json':
                data = pd.read_json(tmp_file_path)
            elif file_ext == '.parquet':
                data = pd.read_parquet(tmp_file_path)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file_ext}"
                )

            # Parse configuration
            if config_json:
                config_dict = json.loads(config_json)
                cleaning_config = CleaningConfig(**config_dict)
            else:
                cleaning_config = self._default_cleaning_config()

            # Submit job
            job_id = await job_manager.submit_job(
                data=data,
                config=cleaning_config,
                filename=file.filename or "uploaded_file",
                original_file_bytes=content,
                original_file_ext=file_ext,
                temp_dir=self.config.temp_dir,
            )

            logger.info(f"Submitted async job {job_id} for file: {file.filename}")
            return {
                "job_id": job_id,
                "status": "submitted",
                "message": "Job submitted successfully",
                "check_status_url": f"/api/jobs/{job_id}"
            }

        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON configuration")
        except Exception as e:
            logger.error(f"Error submitting job: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    async def _clean_data_handler(
        self, 
        file: UploadFile, 
        config_json: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle data cleaning request."""
        # Basic checks
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")

        file_ext = os.path.splitext((file.filename or ""))[1].lower()
        if file_ext not in self.config.allowed_file_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {self.config.allowed_file_types}"
            )

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            if self.config.max_file_size and len(content) > self.config.max_file_size:
                tmp_file.close()
                os.unlink(tmp_file.name)
                raise HTTPException(status_code=413, detail="File too large")
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # Validate file safety with MB limit
            try:
                max_mb = (self.config.max_file_size or (100 * 1024 * 1024)) / (1024 * 1024)
            except Exception:
                max_mb = 100
            validate_file_safety(tmp_file_path, self.config.allowed_file_types)

            # Load data
            if file_ext == '.csv':
                encoding = detect_encoding(tmp_file_path)
                data = pd.read_csv(tmp_file_path, encoding=encoding)
            elif file_ext in ['.xlsx', '.xls']:
                data = pd.read_excel(tmp_file_path)
            elif file_ext == '.json':
                data = pd.read_json(tmp_file_path)
            elif file_ext == '.parquet':
                data = pd.read_parquet(tmp_file_path)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file_ext}"
                )

            # Parse configuration
            if config_json:
                import json
                config_dict = json.loads(config_json)
                cleaning_config = CleaningConfig(**config_dict)
            else:
                cleaning_config = self._default_cleaning_config()

            # Clean data
            cleaned_data, report = clean_data(data, cleaning_config)

            # Build JSON-safe payload
            preview_records = cleaned_data.head(10).to_dict(orient="records")
            preview_sanitized = [
                {k: CleaningAPI._to_jsonable(v) for k, v in row.items()}
                for row in preview_records
            ]

            missing_values_raw = cleaned_data.isna().sum().to_dict()
            missing_values = {
                str(k): CleaningAPI._to_jsonable(v) for k, v in missing_values_raw.items()
            }

            response = {
                "status": "success",
                "original_shape": CleaningAPI._to_jsonable(data.shape),
                "cleaned_shape": CleaningAPI._to_jsonable(cleaned_data.shape),
                "report_summary": CleaningAPI._to_jsonable(report.summary()),
                "preview": preview_sanitized,
                "column_info": {
                    "original_columns": CleaningAPI._to_jsonable(list(data.columns)),
                    "cleaned_columns": CleaningAPI._to_jsonable(list(cleaned_data.columns)),
                    "missing_values": missing_values,
                },
            }

            # Final pass to ensure no NaN/NaT or non-JSON types leak through
            response = CleaningAPI._to_jsonable(response)

            logger.info(f"Successfully cleaned data file: {file.filename}")
            return response

        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)


def create_app(config: Optional[WebConfig] = None) -> FastAPI:
    """
    Create FastAPI application instance.
    
    Parameters
    ----------
    config : WebConfig, optional
        Web application configuration
        
    Returns
    -------
    FastAPI
        Configured FastAPI application
    """
    api = CleaningAPI(config)
    return api.app


# For direct import
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    # Run development server
    uvicorn.run(
        "cleanepi.web.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )