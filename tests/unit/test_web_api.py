"""
Comprehensive tests for the Web API module.
"""

import asyncio
import json
import os
import tempfile
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

from cleanepi.core.config import CleaningConfig, WebConfig
from cleanepi.web.api import CleaningAPI, create_app


@pytest.fixture
def web_config():
    """Create web configuration for testing."""
    return WebConfig(
        max_file_size=1024 * 1024,  # 1MB
        allowed_file_types=[".csv", ".xlsx"],
        temp_dir="/tmp/cleanepi_test",
        enable_async=True,
        chunk_size=1000,
    )


@pytest.fixture
def api_instance(web_config):
    """Create CleaningAPI instance for testing."""
    return CleaningAPI(web_config)


@pytest.fixture
def test_client(api_instance):
    """Create test client for the API."""
    return TestClient(api_instance.app)


@pytest.fixture
def sample_csv_content():
    """Sample CSV content for testing."""
    return "id,name,age\n1,Alice,25\n2,Bob,30\n3,Charlie,35\n"


@pytest.fixture
def sample_excel_file():
    """Create sample Excel file for testing."""
    df = pd.DataFrame(
        {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
    )

    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        df.to_excel(f.name, index=False)
        return f.name


class TestCleaningAPI:
    """Test CleaningAPI class functionality."""

    def test_api_initialization(self, web_config):
        """Test API initialization."""
        api = CleaningAPI(web_config)

        assert api.config == web_config
        assert api.app is not None
        assert api.app.title == "cleanepi API"
        assert api.app.version == "0.1.0"

    def test_api_initialization_default_config(self):
        """Test API initialization with default config."""
        api = CleaningAPI()

        assert isinstance(api.config, WebConfig)
        assert api.app is not None

    def test_health_check_endpoint(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"

    def test_default_config_endpoint(self, test_client):
        """Test default config endpoint."""
        response = test_client.get("/config/default")

        assert response.status_code == 200
        data = response.json()

        # Should return CleaningConfig dict
        assert isinstance(data, dict)
        assert "standardize_column_names" in data
        assert "verbose" in data

    def test_async_endpoint_not_implemented(self, test_client, sample_csv_content):
        """Test async cleaning endpoint (not yet implemented)."""
        csv_file = BytesIO(sample_csv_content.encode())

        response = test_client.post(
            "/clean/async", files={"file": ("test.csv", csv_file, "text/csv")}
        )

        assert response.status_code == 200
        data = response.json()
        assert "not yet implemented" in data["message"]


class TestCleanDataHandler:
    """Test _clean_data_handler functionality."""

    @pytest.mark.asyncio
    async def test_successful_csv_cleaning(self, api_instance, sample_csv_content):
        """Test successful CSV file cleaning."""
        # Create mock upload file
        file_content = sample_csv_content.encode()
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "test.csv"
        upload_file.size = len(file_content)
        upload_file.read = AsyncMock(return_value=file_content)

        # Mock the clean_data function
        mock_data = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
        mock_cleaned = pd.DataFrame({"id": [1, 2], "name": ["alice", "bob"]})
        mock_report = MagicMock()
        mock_report.summary.return_value = {"test": "summary"}

        with patch(
            "cleanepi.web.api.clean_data", return_value=(mock_cleaned, mock_report)
        ):
            with patch("cleanepi.web.api.validate_file_safety"):
                with patch("cleanepi.web.api.detect_encoding", return_value="utf-8"):
                    with patch("pandas.read_csv", return_value=mock_data):
                        result = await api_instance._clean_data_handler(upload_file)

        assert result["status"] == "success"
        assert result["original_shape"] == mock_data.shape
        assert result["cleaned_shape"] == mock_cleaned.shape
        assert "preview" in result
        assert "column_info" in result

    @pytest.mark.asyncio
    async def test_file_too_large_error(self, api_instance):
        """Test file too large error."""
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "large.csv"
        upload_file.size = api_instance.config.max_file_size + 1

        with pytest.raises(Exception):  # HTTPException
            await api_instance._clean_data_handler(upload_file)

    @pytest.mark.asyncio
    async def test_unsupported_file_type_error(self, api_instance):
        """Test unsupported file type error."""
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "test.txt"
        upload_file.size = 100

        with pytest.raises(Exception):  # HTTPException
            await api_instance._clean_data_handler(upload_file)

    @pytest.mark.asyncio
    async def test_with_custom_config(self, api_instance, sample_csv_content):
        """Test cleaning with custom configuration."""
        file_content = sample_csv_content.encode()
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "test.csv"
        upload_file.size = len(file_content)
        upload_file.read = AsyncMock(return_value=file_content)

        config_json = json.dumps({"standardize_column_names": True, "verbose": True})

        mock_data = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        mock_cleaned = pd.DataFrame({"id": [1, 2], "name": ["alice", "bob"]})
        mock_report = MagicMock()
        mock_report.summary.return_value = {"test": "summary"}

        with patch(
            "cleanepi.web.api.clean_data", return_value=(mock_cleaned, mock_report)
        ) as mock_clean:
            with patch("cleanepi.web.api.validate_file_safety"):
                with patch("cleanepi.web.api.detect_encoding", return_value="utf-8"):
                    with patch("pandas.read_csv", return_value=mock_data):
                        result = await api_instance._clean_data_handler(
                            upload_file, config_json
                        )

        # Should use custom config
        mock_clean.assert_called_once()
        config_used = mock_clean.call_args[0][1]
        assert isinstance(config_used, CleaningConfig)
        assert config_used.standardize_column_names is True
        assert config_used.verbose is True

    @pytest.mark.asyncio
    async def test_excel_file_cleaning(self, api_instance):
        """Test Excel file cleaning."""
        # Create Excel content
        df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        excel_buffer = BytesIO()
        df.to_excel(excel_buffer, index=False)
        excel_content = excel_buffer.getvalue()

        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "test.xlsx"
        upload_file.size = len(excel_content)
        upload_file.read = AsyncMock(return_value=excel_content)

        mock_cleaned = pd.DataFrame({"id": [1, 2], "name": ["alice", "bob"]})
        mock_report = MagicMock()
        mock_report.summary.return_value = {"test": "summary"}

        with patch(
            "cleanepi.web.api.clean_data", return_value=(mock_cleaned, mock_report)
        ):
            with patch("cleanepi.web.api.validate_file_safety"):
                with patch("pandas.read_excel", return_value=df):
                    result = await api_instance._clean_data_handler(upload_file)

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_cleaning_error_handling(self, api_instance, sample_csv_content):
        """Test error handling during cleaning."""
        file_content = sample_csv_content.encode()
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "test.csv"
        upload_file.size = len(file_content)
        upload_file.read = AsyncMock(return_value=file_content)

        mock_data = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})

        # Mock clean_data to raise an error
        with patch(
            "cleanepi.web.api.clean_data", side_effect=ValueError("Cleaning failed")
        ):
            with patch("cleanepi.web.api.validate_file_safety"):
                with patch("cleanepi.web.api.detect_encoding", return_value="utf-8"):
                    with patch("pandas.read_csv", return_value=mock_data):
                        with pytest.raises(Exception):  # HTTPException
                            await api_instance._clean_data_handler(upload_file)

    @pytest.mark.asyncio
    async def test_temporary_file_cleanup(self, api_instance, sample_csv_content):
        """Test that temporary files are cleaned up."""
        file_content = sample_csv_content.encode()
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "test.csv"
        upload_file.size = len(file_content)
        upload_file.read = AsyncMock(return_value=file_content)

        mock_data = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        mock_cleaned = pd.DataFrame({"id": [1, 2], "name": ["alice", "bob"]})
        mock_report = MagicMock()
        mock_report.summary.return_value = {"test": "summary"}

        with patch(
            "cleanepi.web.api.clean_data", return_value=(mock_cleaned, mock_report)
        ):
            with patch("cleanepi.web.api.validate_file_safety"):
                with patch("cleanepi.web.api.detect_encoding", return_value="utf-8"):
                    with patch("pandas.read_csv", return_value=mock_data):
                        with patch("os.unlink") as mock_unlink:
                            with patch("os.path.exists", return_value=True):
                                await api_instance._clean_data_handler(upload_file)

                                # Should clean up temp file
                                mock_unlink.assert_called_once()


class TestCreateApp:
    """Test create_app function."""

    def test_create_app_default_config(self):
        """Test creating app with default config."""
        app = create_app()

        assert app is not None
        assert app.title == "cleanepi API"

    def test_create_app_custom_config(self, web_config):
        """Test creating app with custom config."""
        app = create_app(web_config)

        assert app is not None
        assert app.title == "cleanepi API"


class TestIntegrationEndpoints:
    """Test API endpoints through test client."""

    def test_clean_endpoint_csv_file(self, test_client, sample_csv_content):
        """Test clean endpoint with CSV file."""
        csv_file = BytesIO(sample_csv_content.encode())

        # Mock the underlying functions
        mock_data = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        mock_cleaned = pd.DataFrame({"id": [1, 2], "name": ["alice", "bob"]})
        mock_report = MagicMock()
        mock_report.summary.return_value = {
            "total_operations": 1,
            "successful_operations": 1,
            "failed_operations": 0,
        }

        with patch(
            "cleanepi.web.api.clean_data", return_value=(mock_cleaned, mock_report)
        ):
            with patch("cleanepi.web.api.validate_file_safety"):
                with patch("cleanepi.web.api.detect_encoding", return_value="utf-8"):
                    with patch("pandas.read_csv", return_value=mock_data):
                        response = test_client.post(
                            "/clean", files={"file": ("test.csv", csv_file, "text/csv")}
                        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "original_shape" in data
        assert "cleaned_shape" in data
        assert "preview" in data
        assert "column_info" in data

    def test_clean_endpoint_with_config(self, test_client, sample_csv_content):
        """Test clean endpoint with configuration."""
        csv_file = BytesIO(sample_csv_content.encode())
        config_json = json.dumps({"standardize_column_names": True, "verbose": False})

        mock_data = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        mock_cleaned = pd.DataFrame({"id": [1, 2], "name": ["alice", "bob"]})
        mock_report = MagicMock()
        mock_report.summary.return_value = {"test": "summary"}

        with patch(
            "cleanepi.web.api.clean_data", return_value=(mock_cleaned, mock_report)
        ):
            with patch("cleanepi.web.api.validate_file_safety"):
                with patch("cleanepi.web.api.detect_encoding", return_value="utf-8"):
                    with patch("pandas.read_csv", return_value=mock_data):
                        response = test_client.post(
                            "/clean",
                            files={"file": ("test.csv", csv_file, "text/csv")},
                            data={"config_json": config_json},
                        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_clean_endpoint_file_too_large(self, test_client):
        """Test clean endpoint with file too large."""
        # Create large content
        large_content = "x" * (2 * 1024 * 1024)  # 2MB
        large_file = BytesIO(large_content.encode())

        response = test_client.post(
            "/clean", files={"file": ("large.csv", large_file, "text/csv")}
        )

        assert response.status_code == 413  # Request Entity Too Large
        assert "too large" in response.json()["detail"]

    def test_clean_endpoint_invalid_file_type(self, test_client):
        """Test clean endpoint with invalid file type."""
        txt_file = BytesIO(b"some text content")

        response = test_client.post(
            "/clean", files={"file": ("test.txt", txt_file, "text/plain")}
        )

        assert response.status_code == 400
        assert "not allowed" in response.json()["detail"]

    def test_clean_endpoint_processing_error(self, test_client, sample_csv_content):
        """Test clean endpoint with processing error."""
        csv_file = BytesIO(sample_csv_content.encode())

        mock_data = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})

        with patch(
            "cleanepi.web.api.clean_data", side_effect=ValueError("Processing error")
        ):
            with patch("cleanepi.web.api.validate_file_safety"):
                with patch("cleanepi.web.api.detect_encoding", return_value="utf-8"):
                    with patch("pandas.read_csv", return_value=mock_data):
                        response = test_client.post(
                            "/clean", files={"file": ("test.csv", csv_file, "text/csv")}
                        )

        assert response.status_code == 500
        assert "Processing error" in response.json()["detail"]

    def test_clean_endpoint_no_file(self, test_client):
        """Test clean endpoint with no file provided."""
        response = test_client.post("/clean")

        assert response.status_code == 422  # Unprocessable Entity


class TestAPIConfiguration:
    """Test API configuration handling."""

    def test_custom_max_file_size(self):
        """Test custom max file size configuration."""
        config = WebConfig(max_file_size=500000)  # 500KB
        api = CleaningAPI(config)

        assert api.config.max_file_size == 500000

    def test_custom_allowed_file_types(self):
        """Test custom allowed file types."""
        config = WebConfig(allowed_file_types=[".csv"])
        api = CleaningAPI(config)

        assert api.config.allowed_file_types == [".csv"]

    def test_custom_temp_dir(self):
        """Test custom temporary directory."""
        config = WebConfig(temp_dir="/custom/temp")
        api = CleaningAPI(config)

        assert api.config.temp_dir == "/custom/temp"


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_invalid_json_config(self, api_instance, sample_csv_content):
        """Test invalid JSON configuration."""
        file_content = sample_csv_content.encode()
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "test.csv"
        upload_file.size = len(file_content)
        upload_file.read = AsyncMock(return_value=file_content)

        invalid_json = "{ invalid json }"

        with pytest.raises(
            Exception
        ):  # Should raise JSON decode error or validation error
            await api_instance._clean_data_handler(upload_file, invalid_json)

    @pytest.mark.asyncio
    async def test_file_read_error(self, api_instance):
        """Test file read error."""
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "test.csv"
        upload_file.size = 100
        upload_file.read = AsyncMock(side_effect=Exception("Read error"))

        with pytest.raises(Exception):
            await api_instance._clean_data_handler(upload_file)


@pytest.fixture
def mock_upload_file():
    """Create mock UploadFile for testing."""

    def _create_mock_file(filename, content, size=None):
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = filename
        upload_file.size = size or len(content)
        upload_file.read = AsyncMock(return_value=content)
        return upload_file

    return _create_mock_file
