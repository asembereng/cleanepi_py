"""Print cleaning reports for pandas DataFrames."""

import pandas as pd
from typing import Optional, Any, Dict
from .utils import get_report


def print_report(data: pd.DataFrame, 
                key: Optional[str] = None,
                print_output: bool = True) -> Optional[Any]:
    """
    Print cleaning report from a DataFrame.
    
    Args:
        data: DataFrame with cleaning report
        key: Specific report key to print. If None, print all reports.
        print_output: Whether to print the output or just return it
        
    Returns:
        Report data for the specified key, or full report if key is None
        
    Examples:
        >>> import pandas as pd
        >>> # Assume data has been processed with cleanepi functions
        >>> print_report(data, "missing_values_replaced_at")
        >>> print_report(data)  # Print all reports
    """
    report = get_report(data)
    
    if not report:
        if print_output:
            print("No cleaning report found for this data.")
        return None
    
    if key:
        if key in report:
            report_data = report[key]
            if print_output:
                print(f"\n=== {key.upper().replace('_', ' ')} ===")
                _print_report_section(key, report_data)
            return report_data
        else:
            if print_output:
                print(f"Report key '{key}' not found.")
                print(f"Available keys: {list(report.keys())}")
            return None
    else:
        # Print all reports
        if print_output:
            print("\n=== CLEANEPI CLEANING REPORT ===")
            for report_key, report_value in report.items():
                print(f"\n--- {report_key.upper().replace('_', ' ')} ---")
                _print_report_section(report_key, report_value)
        return report


def _print_report_section(key: str, data: Any) -> None:
    """
    Print a specific section of the report.
    
    Args:
        key: Report key
        data: Report data to print
    """
    if key == "missing_values_replaced_at":
        if isinstance(data, list) and data:
            print(f"Missing values were replaced in {len(data)} columns:")
            for col in data:
                print(f"  - {col}")
        else:
            print("No missing values were replaced.")
    
    elif key == "removed_duplicates":
        if isinstance(data, list) and data:
            print(f"Removed {len(data)} duplicate rows:")
            for i, dup in enumerate(data[:5]):  # Show first 5
                print(f"  Row {i+1}: {dup}")
            if len(data) > 5:
                print(f"  ... and {len(data) - 5} more")
        else:
            print("No duplicates were removed.")
    
    elif key == "found_duplicates":
        if isinstance(data, dict):
            dup_rows = data.get("duplicated_rows", [])
            checked_cols = data.get("duplicates_checked_from", [])
            
            print(f"Found {len(dup_rows)} duplicated rows")
            if checked_cols:
                print(f"Checked columns: {', '.join(checked_cols)}")
            
            for i, dup in enumerate(dup_rows[:5]):  # Show first 5
                print(f"  Duplicate {i+1}: {dup}")
            if len(dup_rows) > 5:
                print(f"  ... and {len(dup_rows) - 5} more")
        else:
            print("No duplicate information available.")
    
    elif key == "constant_data":
        if isinstance(data, list) and data:
            print(f"Constant data removal performed in {len(data)} iterations:")
            for iteration in data:
                iter_num = iteration.get('iteration', '?')
                print(f"  Iteration {iter_num}:")
                if iteration.get('empty_columns'):
                    print(f"    Empty columns: {iteration['empty_columns']}")
                if iteration.get('empty_rows'):
                    print(f"    Empty rows: {iteration['empty_rows']}")
                if iteration.get('constant_columns'):
                    print(f"    Constant columns: {iteration['constant_columns']}")
        else:
            print("No constant data was removed.")
    
    elif key == "standardized_dates":
        if isinstance(data, list) and data:
            print(f"Standardized {len(data)} date columns:")
            for col in data:
                print(f"  - {col}")
        else:
            print("No date columns were standardized.")
    
    elif key == "converted_into_numeric":
        if isinstance(data, list) and data:
            print(f"Converted {len(data)} columns to numeric:")
            for col in data:
                print(f"  - {col}")
        else:
            print("No columns were converted to numeric.")
    
    elif key == "subject_id_check":
        if isinstance(data, dict):
            missing = data.get("missing_ids", [])
            duplicates = data.get("duplicate_ids", [])
            incorrect = data.get("incorrect_ids", [])
            
            print(f"Subject ID check results:")
            print(f"  Missing IDs: {len(missing)}")
            print(f"  Duplicate IDs: {len(duplicates)}")
            print(f"  Incorrect format IDs: {len(incorrect)}")
            
            if incorrect:
                print("  Incorrect IDs:")
                for col, idx, val in incorrect[:5]:
                    print(f"    {col}[{idx}]: {val}")
                if len(incorrect) > 5:
                    print(f"    ... and {len(incorrect) - 5} more")
        else:
            print("No subject ID check information available.")
    
    elif key == "date_guesses":
        if isinstance(data, dict) and data:
            print(f"Date format guesses for {len(data)} columns:")
            for col, guess_info in data.items():
                confidence = guess_info.get('confidence', 0)
                formats = guess_info.get('formats', [])
                print(f"  {col}: {confidence:.1%} confidence")
                if formats:
                    print(f"    Likely formats: {', '.join(formats)}")
        else:
            print("No date guesses available.")
    
    else:
        # Generic printing for unknown report types
        if isinstance(data, (list, dict)):
            if len(str(data)) < 500:  # Print short data directly
                print(data)
            else:
                print(f"<{type(data).__name__} with {len(data)} items>")
        else:
            print(data)


def get_cleaning_summary(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Get a summary of all cleaning operations performed.
    
    Args:
        data: DataFrame with cleaning report
        
    Returns:
        Dictionary summarizing cleaning operations
    """
    report = get_report(data)
    
    summary = {
        'operations_performed': list(report.keys()) if report else [],
        'total_operations': len(report) if report else 0
    }
    
    if report:
        # Count specific operations
        if 'missing_values_replaced_at' in report:
            summary['columns_with_missing_values_replaced'] = len(report['missing_values_replaced_at'])
        
        if 'removed_duplicates' in report:
            summary['duplicates_removed'] = len(report['removed_duplicates'])
        
        if 'standardized_dates' in report:
            summary['date_columns_standardized'] = len(report['standardized_dates'])
        
        if 'converted_into_numeric' in report:
            summary['columns_converted_to_numeric'] = len(report['converted_into_numeric'])
    
    return summary