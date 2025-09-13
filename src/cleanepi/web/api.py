"""
FastAPI web application components for cleanepi.

This module provides REST API endpoints for data cleaning operations.
"""

import os
import tempfile
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger

from ..core.clean_data import clean_data
from ..core.config import CleaningConfig, WebConfig
from ..utils.validation import detect_encoding, validate_file_safety


class CleaningAPI:
    """FastAPI application for data cleaning operations."""

    def __init__(self, config: Optional[WebConfig] = None):
        """Initialize the cleaning API."""
        self.config = config or WebConfig()
        self.app = FastAPI(
            title="cleanepi API",
            description="Clean and standardize epidemiological data",
            version="0.1.0",
        )
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.post("/clean")
        async def clean_data_endpoint(
            file: UploadFile = File(...), config_json: Optional[str] = None
        ):
            """
            Clean uploaded data file.

            Parameters:
            - file: CSV or Excel file to clean
            - config_json: Optional JSON string with cleaning configuration
            """
            return await self._clean_data_handler(file, config_json)

        @self.app.post("/clean/async")
        async def clean_data_async(
            background_tasks: BackgroundTasks,
            file: UploadFile = File(...),
            config_json: Optional[str] = None,
        ):
            """
            Start async data cleaning job.

            Returns job ID for status checking.
            """
            # TODO: Implement async job processing
            return {"message": "Async processing not yet implemented"}

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "version": "0.1.0"}

        @self.app.get("/config/default")
        async def get_default_config():
            """Get default cleaning configuration."""
            config = CleaningConfig()
            return config.dict()

    async def _clean_data_handler(
        self, file: UploadFile, config_json: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle data cleaning request."""

        # Validate file
        if file.size > self.config.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {self.config.max_file_size} bytes",
            )

        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in self.config.allowed_file_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {self.config.allowed_file_types}",
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
            if file_ext == ".csv":
                encoding = detect_encoding(tmp_file_path)
                data = pd.read_csv(tmp_file_path, encoding=encoding)
            elif file_ext in [".xlsx", ".xls"]:
                data = pd.read_excel(tmp_file_path)
            else:
                raise HTTPException(
                    status_code=400, detail=f"Unsupported file type: {file_ext}"
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
                    "missing_values": cleaned_data.isna().sum().to_dict(),
                },
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
        "cleanepi.web.api:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
