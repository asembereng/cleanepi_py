"""
Comprehensive tests for the report module.
"""

import pytest
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import patch, mock_open
import pandas as pd

from cleanepi.core.report import OperationResult, CleaningReport


class TestOperationResult:
    """Test OperationResult functionality."""
    
    def test_basic_creation(self):
        """Test basic OperationResult creation."""
        timestamp = datetime.now()
        result = OperationResult(
            operation="test_operation",
            timestamp=timestamp,
            success=True,
            rows_before=100,
            rows_after=95,
            columns_before=10,
            columns_after=9
        )
        
        assert result.operation == "test_operation"
        assert result.timestamp == timestamp
        assert result.success is True
        assert result.rows_before == 100
        assert result.rows_after == 95
        assert result.columns_before == 10
        assert result.columns_after == 9
        assert result.details == {}
        assert result.warnings == []
        assert result.errors == []
    
    def test_with_details_warnings_errors(self):
        """Test OperationResult with additional data."""
        timestamp = datetime.now()
        details = {"removed_columns": ["col1", "col2"]}
        warnings = ["Warning 1", "Warning 2"]
        errors = ["Error 1"]
        
        result = OperationResult(
            operation="remove_constants",
            timestamp=timestamp,
            success=False,
            rows_before=100,
            rows_after=100,
            columns_before=10,
            columns_after=8,
            details=details,
            warnings=warnings,
            errors=errors
        )
        
        assert result.details == details
        assert result.warnings == warnings
        assert result.errors == errors
    
    def test_rows_changed_property(self):
        """Test rows_changed property calculation."""
        # Rows decreased
        result = OperationResult(
            operation="test",
            timestamp=datetime.now(),
            success=True,
            rows_before=100,
            rows_after=90,
            columns_before=5,
            columns_after=5
        )
        assert result.rows_changed == 10
        
        # Rows increased (shouldn't happen in cleaning but test anyway)
        result = OperationResult(
            operation="test",
            timestamp=datetime.now(),
            success=True,
            rows_before=100,
            rows_after=110,
            columns_before=5,
            columns_after=5
        )
        assert result.rows_changed == 10
        
        # No change
        result = OperationResult(
            operation="test",
            timestamp=datetime.now(),
            success=True,
            rows_before=100,
            rows_after=100,
            columns_before=5,
            columns_after=5
        )
        assert result.rows_changed == 0
    
    def test_columns_changed_property(self):
        """Test columns_changed property calculation."""
        # Columns decreased
        result = OperationResult(
            operation="test",
            timestamp=datetime.now(),
            success=True,
            rows_before=100,
            rows_after=100,
            columns_before=10,
            columns_after=8
        )
        assert result.columns_changed == 2
        
        # Columns increased (shouldn't happen in cleaning but test anyway)
        result = OperationResult(
            operation="test",
            timestamp=datetime.now(),
            success=True,
            rows_before=100,
            rows_after=100,
            columns_before=10,
            columns_after=12
        )
        assert result.columns_changed == 2
        
        # No change
        result = OperationResult(
            operation="test",
            timestamp=datetime.now(),
            success=True,
            rows_before=100,
            rows_after=100,
            columns_before=10,
            columns_after=10
        )
        assert result.columns_changed == 0


class TestCleaningReport:
    """Test CleaningReport functionality."""
    
    def test_initialization(self):
        """Test CleaningReport initialization."""
        initial_shape = (100, 10)
        report = CleaningReport(initial_shape)
        
        assert report.initial_shape == initial_shape
        assert isinstance(report.start_time, datetime)
        assert report.end_time is None
        assert report.operations == []
        assert report.metadata == {}
    
    def test_add_operation(self):
        """Test adding operations to report."""
        report = CleaningReport((100, 10))
        
        result = OperationResult(
            operation="test_op",
            timestamp=datetime.now(),
            success=True,
            rows_before=100,
            rows_after=95,
            columns_before=10,
            columns_after=9
        )
        
        with patch('cleanepi.core.report.logger') as mock_logger:
            report.add_operation(result)
            
            assert len(report.operations) == 1
            assert report.operations[0] == result
            mock_logger.info.assert_called_once()
    
    def test_finalize(self):
        """Test report finalization."""
        report = CleaningReport((100, 10))
        final_shape = (95, 8)
        
        # Test finalization
        report.finalize(final_shape)
        
        assert report.final_shape == final_shape
        assert isinstance(report.end_time, datetime)
        assert report.end_time > report.start_time
    
    def test_duration_property(self):
        """Test duration property."""
        report = CleaningReport((100, 10))
        
        # Before finalization
        assert report.duration is None
        
        # After finalization
        report.finalize((95, 8))
        assert isinstance(report.duration, float)
        assert report.duration >= 0
    
    def test_total_rows_removed(self):
        """Test total_rows_removed property."""
        report = CleaningReport((100, 10))
        
        # Before finalization
        assert report.total_rows_removed == 0
        
        # After finalization
        report.finalize((95, 8))
        assert report.total_rows_removed == 5
        
        # Test with no rows removed
        report2 = CleaningReport((100, 10))
        report2.finalize((100, 8))
        assert report2.total_rows_removed == 0
    
    def test_total_columns_removed(self):
        """Test total_columns_removed property."""
        report = CleaningReport((100, 10))
        
        # Before finalization
        assert report.total_columns_removed == 0
        
        # After finalization
        report.finalize((95, 8))
        assert report.total_columns_removed == 2
        
        # Test with no columns removed
        report2 = CleaningReport((100, 10))
        report2.finalize((95, 10))
        assert report2.total_columns_removed == 0
    
    def test_successful_operations(self):
        """Test successful_operations property."""
        report = CleaningReport((100, 10))
        
        # Add successful operation
        success_result = OperationResult(
            operation="success_op",
            timestamp=datetime.now(),
            success=True,
            rows_before=100,
            rows_after=95,
            columns_before=10,
            columns_after=9
        )
        report.add_operation(success_result)
        
        # Add failed operation
        fail_result = OperationResult(
            operation="fail_op",
            timestamp=datetime.now(),
            success=False,
            rows_before=95,
            rows_after=95,
            columns_before=9,
            columns_after=9,
            errors=["Test error"]
        )
        report.add_operation(fail_result)
        
        successful = report.successful_operations
        assert len(successful) == 1
        assert successful[0] == success_result
    
    def test_failed_operations(self):
        """Test failed_operations property."""
        report = CleaningReport((100, 10))
        
        # Add successful operation
        success_result = OperationResult(
            operation="success_op",
            timestamp=datetime.now(),
            success=True,
            rows_before=100,
            rows_after=95,
            columns_before=10,
            columns_after=9
        )
        report.add_operation(success_result)
        
        # Add failed operation
        fail_result = OperationResult(
            operation="fail_op",
            timestamp=datetime.now(),
            success=False,
            rows_before=95,
            rows_after=95,
            columns_before=9,
            columns_after=9,
            errors=["Test error"]
        )
        report.add_operation(fail_result)
        
        failed = report.failed_operations
        assert len(failed) == 1
        assert failed[0] == fail_result
    
    def test_all_warnings(self):
        """Test all_warnings property."""
        report = CleaningReport((100, 10))
        
        # Add operations with warnings
        result1 = OperationResult(
            operation="op1",
            timestamp=datetime.now(),
            success=True,
            rows_before=100,
            rows_after=95,
            columns_before=10,
            columns_after=9,
            warnings=["Warning 1", "Warning 2"]
        )
        report.add_operation(result1)
        
        result2 = OperationResult(
            operation="op2",
            timestamp=datetime.now(),
            success=True,
            rows_before=95,
            rows_after=90,
            columns_before=9,
            columns_after=8,
            warnings=["Warning 3"]
        )
        report.add_operation(result2)
        
        all_warnings = report.all_warnings
        assert len(all_warnings) == 3
        assert "Warning 1" in all_warnings
        assert "Warning 2" in all_warnings
        assert "Warning 3" in all_warnings
    
    def test_all_errors(self):
        """Test all_errors property."""
        report = CleaningReport((100, 10))
        
        # Add operations with errors
        result1 = OperationResult(
            operation="op1",
            timestamp=datetime.now(),
            success=False,
            rows_before=100,
            rows_after=100,
            columns_before=10,
            columns_after=10,
            errors=["Error 1", "Error 2"]
        )
        report.add_operation(result1)
        
        result2 = OperationResult(
            operation="op2",
            timestamp=datetime.now(),
            success=False,
            rows_before=100,
            rows_after=100,
            columns_before=10,
            columns_after=10,
            errors=["Error 3"]
        )
        report.add_operation(result2)
        
        all_errors = report.all_errors
        assert len(all_errors) == 3
        assert "Error 1" in all_errors
        assert "Error 2" in all_errors
        assert "Error 3" in all_errors
    
    def test_summary(self):
        """Test summary generation."""
        report = CleaningReport((100, 10))
        
        # Add an operation
        result = OperationResult(
            operation="test_op",
            timestamp=datetime.now(),
            success=True,
            rows_before=100,
            rows_after=95,
            columns_before=10,
            columns_after=9,
            warnings=["Test warning"]
        )
        report.add_operation(result)
        
        # Finalize report
        report.finalize((95, 9))
        
        summary = report.summary()
        
        assert "start_time" in summary
        assert "end_time" in summary
        assert "duration_seconds" in summary
        assert summary["initial_shape"] == (100, 10)
        assert summary["final_shape"] == (95, 9)
        assert summary["total_operations"] == 1
        assert summary["successful_operations"] == 1
        assert summary["failed_operations"] == 0
        assert summary["total_rows_removed"] == 5
        assert summary["total_columns_removed"] == 1
        assert summary["total_warnings"] == 1
        assert summary["total_errors"] == 0
        assert "operations" in summary
        assert len(summary["operations"]) == 1
    
    def test_detailed_report(self):
        """Test detailed report generation."""
        report = CleaningReport((100, 10))
        
        # Add an operation
        result = OperationResult(
            operation="test_op",
            timestamp=datetime.now(),
            success=True,
            rows_before=100,
            rows_after=95,
            columns_before=10,
            columns_after=9,
            details={"test_detail": "value"},
            warnings=["Test warning"],
            errors=[]
        )
        report.add_operation(result)
        
        # Add metadata
        report.metadata = {"test_meta": "meta_value"}
        
        # Finalize report
        report.finalize((95, 9))
        
        detailed = report.detailed_report()
        
        # Should contain all summary fields
        assert "start_time" in detailed
        assert "operations" in detailed
        
        # Should contain detailed operations
        assert "detailed_operations" in detailed
        assert len(detailed["detailed_operations"]) == 1
        
        detailed_op = detailed["detailed_operations"][0]
        assert detailed_op["operation"] == "test_op"
        assert detailed_op["success"] is True
        assert detailed_op["shape_before"] == (100, 10)
        assert detailed_op["shape_after"] == (95, 9)
        assert detailed_op["details"] == {"test_detail": "value"}
        assert detailed_op["warnings"] == ["Test warning"]
        assert detailed_op["errors"] == []
        
        # Should contain metadata
        assert "metadata" in detailed
        assert detailed["metadata"] == {"test_meta": "meta_value"}
    
    def test_to_json(self):
        """Test JSON export."""
        report = CleaningReport((100, 10))
        
        # Add operation and finalize
        result = OperationResult(
            operation="test_op",
            timestamp=datetime.now(),
            success=True,
            rows_before=100,
            rows_after=95,
            columns_before=10,
            columns_after=9
        )
        report.add_operation(result)
        report.finalize((95, 9))
        
        json_str = report.to_json()
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "start_time" in parsed
        assert "detailed_operations" in parsed
        
        # Test custom indent
        json_str_compact = report.to_json(indent=0)
        assert len(json_str_compact) <= len(json_str)  # Should be more compact
    
    def test_to_file_json(self):
        """Test export to JSON file."""
        report = CleaningReport((100, 10))
        report.finalize((95, 9))
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            report.to_file(temp_path, format="json")
            
            # Verify file was created and contains valid JSON
            assert os.path.exists(temp_path)
            with open(temp_path, 'r') as f:
                content = f.read()
                parsed = json.loads(content)
                assert isinstance(parsed, dict)
        finally:
            os.unlink(temp_path)
    
    def test_to_file_unsupported_format(self):
        """Test export with unsupported format."""
        report = CleaningReport((100, 10))
        
        with pytest.raises(ValueError, match="Unsupported format"):
            report.to_file("test.xml", format="xml")
    
    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        report = CleaningReport((100, 10))
        
        # Add multiple operations
        result1 = OperationResult(
            operation="op1",
            timestamp=datetime.now(),
            success=True,
            rows_before=100,
            rows_after=95,
            columns_before=10,
            columns_after=9,
            warnings=["Warning 1"]
        )
        report.add_operation(result1)
        
        result2 = OperationResult(
            operation="op2",
            timestamp=datetime.now(),
            success=False,
            rows_before=95,
            rows_after=95,
            columns_before=9,
            columns_after=9,
            errors=["Error 1", "Error 2"]
        )
        report.add_operation(result2)
        
        df = report.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        
        expected_columns = [
            "operation", "timestamp", "success", "rows_before", "rows_after",
            "columns_before", "columns_after", "rows_changed", "columns_changed",
            "warnings_count", "errors_count"
        ]
        for col in expected_columns:
            assert col in df.columns
        
        # Check first row
        assert df.iloc[0]["operation"] == "op1"
        assert df.iloc[0]["success"] is True
        assert df.iloc[0]["rows_changed"] == 5
        assert df.iloc[0]["columns_changed"] == 1
        assert df.iloc[0]["warnings_count"] == 1
        assert df.iloc[0]["errors_count"] == 0
        
        # Check second row
        assert df.iloc[1]["operation"] == "op2"
        assert df.iloc[1]["success"] is False
        assert df.iloc[1]["rows_changed"] == 0
        assert df.iloc[1]["columns_changed"] == 0
        assert df.iloc[1]["warnings_count"] == 0
        assert df.iloc[1]["errors_count"] == 2
    
    def test_to_dataframe_empty(self):
        """Test conversion to DataFrame with no operations."""
        report = CleaningReport((100, 10))
        df = report.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert len(df.columns) > 0  # Should have column structure even if empty
    
    @patch('builtins.print')
    def test_print_summary(self, mock_print):
        """Test print_summary method."""
        report = CleaningReport((100, 10))
        
        # Add operations
        result1 = OperationResult(
            operation="op1",
            timestamp=datetime.now(),
            success=True,
            rows_before=100,
            rows_after=95,
            columns_before=10,
            columns_after=9,
            warnings=["Warning 1"]
        )
        report.add_operation(result1)
        
        result2 = OperationResult(
            operation="op2",
            timestamp=datetime.now(),
            success=False,
            rows_before=95,
            rows_after=95,
            columns_before=9,
            columns_after=9,
            errors=["Error 1"]
        )
        report.add_operation(result2)
        
        report.finalize((95, 9))
        
        report.print_summary()
        
        # Verify print was called multiple times
        assert mock_print.call_count > 0
        
        # Check some expected content was printed
        printed_text = " ".join([str(call.args[0]) for call in mock_print.call_args_list])
        assert "DATA CLEANING REPORT SUMMARY" in printed_text
        assert "Initial:" in printed_text
        assert "Final:" in printed_text
        assert "Total:" in printed_text
    
    def test_get_operation_details(self):
        """Test get_operation_details method."""
        report = CleaningReport((100, 10))
        
        result = OperationResult(
            operation="test_operation",
            timestamp=datetime.now(),
            success=True,
            rows_before=100,
            rows_after=95,
            columns_before=10,
            columns_after=9
        )
        report.add_operation(result)
        
        # Test existing operation
        found_result = report.get_operation_details("test_operation")
        assert found_result == result
        
        # Test non-existing operation
        not_found = report.get_operation_details("nonexistent_operation")
        assert not_found is None
    
    def test_has_errors(self):
        """Test has_errors method."""
        report = CleaningReport((100, 10))
        
        # No errors initially
        assert report.has_errors() is False
        
        # Add successful operation
        success_result = OperationResult(
            operation="success_op",
            timestamp=datetime.now(),
            success=True,
            rows_before=100,
            rows_after=95,
            columns_before=10,
            columns_after=9
        )
        report.add_operation(success_result)
        assert report.has_errors() is False
        
        # Add failed operation
        fail_result = OperationResult(
            operation="fail_op",
            timestamp=datetime.now(),
            success=False,
            rows_before=95,
            rows_after=95,
            columns_before=9,
            columns_after=9,
            errors=["Test error"]
        )
        report.add_operation(fail_result)
        assert report.has_errors() is True
    
    def test_has_warnings(self):
        """Test has_warnings method."""
        report = CleaningReport((100, 10))
        
        # No warnings initially
        assert report.has_warnings() is False
        
        # Add operation without warnings
        no_warning_result = OperationResult(
            operation="no_warning_op",
            timestamp=datetime.now(),
            success=True,
            rows_before=100,
            rows_after=95,
            columns_before=10,
            columns_after=9
        )
        report.add_operation(no_warning_result)
        assert report.has_warnings() is False
        
        # Add operation with warnings
        warning_result = OperationResult(
            operation="warning_op",
            timestamp=datetime.now(),
            success=True,
            rows_before=95,
            rows_after=90,
            columns_before=9,
            columns_after=8,
            warnings=["Test warning"]
        )
        report.add_operation(warning_result)
        assert report.has_warnings() is True


@pytest.fixture
def sample_operation_result():
    """Create a sample OperationResult for testing."""
    return OperationResult(
        operation="test_operation",
        timestamp=datetime.now(),
        success=True,
        rows_before=100,
        rows_after=95,
        columns_before=10,
        columns_after=9,
        details={"test": "value"},
        warnings=["Test warning"],
        errors=[]
    )


@pytest.fixture
def sample_cleaning_report():
    """Create a sample CleaningReport for testing."""
    report = CleaningReport((100, 10))
    
    result = OperationResult(
        operation="test_operation",
        timestamp=datetime.now(),
        success=True,
        rows_before=100,
        rows_after=95,
        columns_before=10,
        columns_after=9
    )
    report.add_operation(result)
    report.finalize((95, 9))
    
    return report