"""
Data cleaning report generation and management.

Provides comprehensive reporting of all cleaning operations performed.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
import json
import pandas as pd
from loguru import logger


@dataclass
class OperationResult:
    """Result of a single cleaning operation."""
    
    operation: str
    timestamp: datetime
    success: bool
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def rows_changed(self) -> int:
        """Number of rows changed."""
        return abs(self.rows_after - self.rows_before)
    
    @property
    def columns_changed(self) -> int:
        """Number of columns changed."""
        return abs(self.columns_after - self.columns_before)


class CleaningReport:
    """Comprehensive report of data cleaning operations."""
    
    def __init__(self, initial_shape: tuple):
        """Initialize report with initial data shape."""
        self.initial_shape = initial_shape
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.operations: List[OperationResult] = []
        self.metadata: Dict[str, Any] = {}
        
    def add_operation(self, result: OperationResult) -> None:
        """Add an operation result to the report."""
        self.operations.append(result)
        logger.info(f"Operation '{result.operation}' completed: "
                   f"{result.rows_before} -> {result.rows_after} rows, "
                   f"{result.columns_before} -> {result.columns_after} columns")
    
    def finalize(self, final_shape: tuple) -> None:
        """Finalize the report with final data shape."""
        self.end_time = datetime.now()
        self.final_shape = final_shape
        
    @property
    def duration(self) -> Optional[float]:
        """Total processing duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def total_rows_removed(self) -> int:
        """Total number of rows removed across all operations."""
        return self.initial_shape[0] - self.final_shape[0] if hasattr(self, 'final_shape') else 0
    
    @property
    def total_columns_removed(self) -> int:
        """Total number of columns removed across all operations."""
        return self.initial_shape[1] - self.final_shape[1] if hasattr(self, 'final_shape') else 0
    
    @property
    def successful_operations(self) -> List[OperationResult]:
        """List of successful operations."""
        return [op for op in self.operations if op.success]
    
    @property
    def failed_operations(self) -> List[OperationResult]:
        """List of failed operations."""
        return [op for op in self.operations if not op.success]
    
    @property
    def all_warnings(self) -> List[str]:
        """All warnings from all operations."""
        warnings = []
        for op in self.operations:
            warnings.extend(op.warnings)
        return warnings
    
    @property
    def all_errors(self) -> List[str]:
        """All errors from all operations."""
        errors = []
        for op in self.operations:
            errors.extend(op.errors)
        return errors
    
    def summary(self) -> Dict[str, Any]:
        """Generate a summary of the cleaning report."""
        summary = {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration,
            "initial_shape": self.initial_shape,
            "final_shape": getattr(self, 'final_shape', None),
            "total_operations": len(self.operations),
            "successful_operations": len(self.successful_operations),
            "failed_operations": len(self.failed_operations),
            "total_rows_removed": self.total_rows_removed,
            "total_columns_removed": self.total_columns_removed,
            "total_warnings": len(self.all_warnings),
            "total_errors": len(self.all_errors),
        }
        
        # Add operation-specific summaries
        operation_summaries = []
        for op in self.operations:
            op_summary = {
                "operation": op.operation,
                "success": op.success,
                "rows_changed": op.rows_changed,
                "columns_changed": op.columns_changed,
                "warnings": len(op.warnings),
                "errors": len(op.errors)
            }
            operation_summaries.append(op_summary)
        
        summary["operations"] = operation_summaries
        return summary
    
    def detailed_report(self) -> Dict[str, Any]:
        """Generate a detailed report including all operation details."""
        detailed = self.summary()
        
        # Add detailed operation results
        detailed_operations = []
        for op in self.operations:
            op_detail = {
                "operation": op.operation,
                "timestamp": op.timestamp.isoformat(),
                "success": op.success,
                "shape_before": (op.rows_before, op.columns_before),
                "shape_after": (op.rows_after, op.columns_after),
                "details": op.details,
                "warnings": op.warnings,
                "errors": op.errors
            }
            detailed_operations.append(op_detail)
        
        detailed["detailed_operations"] = detailed_operations
        detailed["metadata"] = self.metadata
        
        return detailed
    
    def to_json(self, indent: int = 2) -> str:
        """Export report as JSON string."""
        return json.dumps(self.detailed_report(), indent=indent, default=str)
    
    def to_file(self, filepath: str, format: str = "json") -> None:
        """Export report to file."""
        if format.lower() == "json":
            with open(filepath, "w") as f:
                f.write(self.to_json())
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert operations to DataFrame for analysis."""
        rows = []
        for op in self.operations:
            row = {
                "operation": op.operation,
                "timestamp": op.timestamp,
                "success": op.success,
                "rows_before": op.rows_before,
                "rows_after": op.rows_after,
                "columns_before": op.columns_before,
                "columns_after": op.columns_after,
                "rows_changed": op.rows_changed,
                "columns_changed": op.columns_changed,
                "warnings_count": len(op.warnings),
                "errors_count": len(op.errors)
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def print_summary(self) -> None:
        """Print a formatted summary of the cleaning report."""
        print("=" * 60)
        print("DATA CLEANING REPORT SUMMARY")
        print("=" * 60)
        
        summary = self.summary()
        
        print(f"Start Time: {summary['start_time']}")
        if summary['end_time']:
            print(f"End Time: {summary['end_time']}")
            print(f"Duration: {summary['duration_seconds']:.2f} seconds")
        
        print(f"\nData Shape:")
        print(f"  Initial: {summary['initial_shape']}")
        if summary['final_shape']:
            print(f"  Final:   {summary['final_shape']}")
            print(f"  Removed: {summary['total_rows_removed']} rows, "
                  f"{summary['total_columns_removed']} columns")
        
        print(f"\nOperations:")
        print(f"  Total:      {summary['total_operations']}")
        print(f"  Successful: {summary['successful_operations']}")
        print(f"  Failed:     {summary['failed_operations']}")
        print(f"  Warnings:   {summary['total_warnings']}")
        print(f"  Errors:     {summary['total_errors']}")
        
        if summary['operations']:
            print(f"\nOperation Details:")
            for op in summary['operations']:
                status = "✓" if op['success'] else "✗"
                print(f"  {status} {op['operation']}: "
                      f"{op['rows_changed']} rows, {op['columns_changed']} columns")
                if op['warnings'] > 0:
                    print(f"    ⚠ {op['warnings']} warnings")
                if op['errors'] > 0:
                    print(f"    ✗ {op['errors']} errors")
        
        print("=" * 60)
    
    def get_operation_details(self, operation_name: str) -> Optional[OperationResult]:
        """Get details for a specific operation."""
        for op in self.operations:
            if op.operation == operation_name:
                return op
        return None
    
    def has_errors(self) -> bool:
        """Check if any operations had errors."""
        return len(self.failed_operations) > 0 or len(self.all_errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if any operations had warnings."""
        return len(self.all_warnings) > 0