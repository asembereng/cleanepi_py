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
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import pandas as pd
from loguru import logger

from ..core.clean_data import clean_data
from ..core.config import CleaningConfig, WebConfig
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
        self.config = config or WebConfig()
        self.app = FastAPI(
            title="cleanepi API",
            description="Clean and standardize epidemiological data",
            version="0.1.0",
            lifespan=lifespan
        )
        
        # Setup templates and static files
        templates_dir = os.path.join(os.path.dirname(__file__), "templates")
        self.templates = Jinja2Templates(directory=templates_dir)
        
        self._setup_routes()
        
        # Mount static files (will be created later)
        try:
            static_dir = os.path.join(os.path.dirname(__file__), "static")
            self.app.mount("/static", StaticFiles(directory=static_dir), name="static")
        except Exception:
            logger.warning("Static files directory not found, skipping static file serving")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        # Web UI Routes
        @self.app.get("/", response_class=HTMLResponse)
        async def web_interface(request: Request):
            """Main web interface."""
            return self.templates.TemplateResponse(
                "index.html",
                {"request": request, "title": "cleanepi - Data Cleaning Tool"}
            )
        
        @self.app.get("/jobs", response_class=HTMLResponse)
        async def jobs_interface(request: Request):
            """Jobs management interface."""
            return self.templates.TemplateResponse(
                "jobs.html", 
                {"request": request, "title": "Job Management"}
            )
        
        @self.app.get("/config", response_class=HTMLResponse)
        async def config_interface(request: Request):
            """Configuration interface."""
            return self.templates.TemplateResponse(
                "config.html",
                {"request": request, "title": "Configuration"}
            )
        
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
            config = CleaningConfig()
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
            config = CleaningConfig()
            return config.dict()
    
    async def _submit_job_handler(
        self,
        file: UploadFile,
        config_json: Optional[str],
        job_manager: JobManager
    ) -> Dict[str, Any]:
        """Handle async job submission."""
        
        # Validate file
        if file.size > self.config.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {self.config.max_file_size} bytes"
            )
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in self.config.allowed_file_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {self.config.allowed_file_types}"
            )
        
        # Create temporary file and load data
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Validate file safety
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
                cleaning_config = CleaningConfig()
            
            # Submit job
            job_id = await job_manager.submit_job(
                data=data,
                config=cleaning_config,
                filename=file.filename or "uploaded_file"
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
        
        # Validate file
        if file.size > self.config.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {self.config.max_file_size} bytes"
            )
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in self.config.allowed_file_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {self.config.allowed_file_types}"
            )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Validate file safety
            validate_file_safety(tmp_file_path, self.config.allowed_file_types)
            
            # Load data
            if file_ext == '.csv':
                encoding = detect_encoding(tmp_file_path)
                data = pd.read_csv(tmp_file_path, encoding=encoding)
            elif file_ext in ['.xlsx', '.xls']:
                data = pd.read_excel(tmp_file_path)
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
                cleaning_config = CleaningConfig()
            
            # Clean data
            cleaned_data, report = clean_data(data, cleaning_config)
            
            # Prepare response
            response = {
                "status": "success",
                "original_shape": data.shape,
                "cleaned_shape": cleaned_data.shape,
                "report_summary": report.summary(),
                "preview": cleaned_data.head(10).to_dict(orient="records"),
                "column_info": {
                    "original_columns": list(data.columns),
                    "cleaned_columns": list(cleaned_data.columns),
                    "missing_values": cleaned_data.isna().sum().to_dict()
                }
            }
            
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