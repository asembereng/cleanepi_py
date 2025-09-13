# Contributing to cleanepi-python

Thank you for your interest in contributing to cleanepi-python! This guide will help you set up the development environment and understand the codebase structure.

## Development Setup

### 1. Clone and Setup

```bash
git clone https://github.com/epiverse-trace/cleanepi.git
cd cleanepi/cleanepi_python

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### 2. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/cleanepi --cov-report=html

# Run specific test
pytest tests/test_basic_functionality.py::TestBasicFunctionality::test_clean_data_basic -v
```

### 3. Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Project Structure

```
cleanepi_python/
├── src/cleanepi/           # Main package source
│   ├── core/              # Core functionality
│   │   ├── clean_data.py  # Main orchestration function
│   │   ├── config.py      # Configuration models (Pydantic)
│   │   └── report.py      # Reporting and logging
│   ├── cleaning/          # Individual cleaning functions
│   │   ├── standardize_columns.py
│   │   ├── replace_missing.py
│   │   ├── remove_duplicates.py
│   │   ├── remove_constants.py
│   │   ├── standardize_dates.py      # TODO: Implement
│   │   ├── validate_subject_ids.py   # TODO: Implement
│   │   ├── convert_numeric.py        # TODO: Implement
│   │   ├── dictionary_cleaning.py    # TODO: Implement
│   │   └── date_sequence.py          # TODO: Implement
│   ├── utils/             # Utility functions
│   │   ├── validation.py  # Input validation and security
│   │   └── data_scanning.py          # TODO: Implement
│   ├── web/               # TODO: Web application components
│   └── async_processing/  # TODO: Async processing for large datasets
├── tests/                 # Test suite
├── examples/              # Usage examples
├── docs/                  # Documentation
└── pyproject.toml         # Project configuration
```

## Design Principles

### 1. **Scalability**
- Use pandas for medium datasets (<1GB)
- Provide Dask integration for larger datasets
- Chunk processing for memory efficiency
- Async support for I/O operations

### 2. **Reliability**
- Comprehensive error handling with graceful degradation
- Extensive validation using Pydantic
- Detailed logging with loguru
- Comprehensive test coverage (>90%)

### 3. **Security**
- Input validation and sanitization
- Safe file operations with path traversal protection
- Memory usage limits
- Protection against code injection

### 4. **Web-Ready Architecture**
- RESTful API design patterns
- Async/await support
- JSON serializable configurations
- Detailed operation reporting

## Adding New Cleaning Functions

### 1. Create the Function File

```python
# src/cleanepi/cleaning/my_new_function.py
from typing import List, Optional
import pandas as pd
from loguru import logger
from ..utils.validation import validate_dataframe

def my_cleaning_function(
    data: pd.DataFrame,
    param1: str,
    param2: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Brief description of what this function does.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
    param1 : str
        Description of param1
    param2 : List[str], optional
        Description of param2
        
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame
    """
    validate_dataframe(data)
    
    # Implementation here
    result = data.copy()
    
    logger.info(f"Applied my_cleaning_function")
    return result
```

### 2. Add Configuration Model

```python
# Add to src/cleanepi/core/config.py
class MyFunctionConfig(BaseModel):
    """Configuration for my cleaning function."""
    
    param1: str = Field(..., description="Required parameter")
    param2: Optional[List[str]] = Field(None, description="Optional parameter")
```

### 3. Update Main Configuration

```python
# In CleaningConfig class
my_function: Optional[MyFunctionConfig] = Field(
    None,
    description="Configuration for my cleaning function"
)
```

### 4. Add to Main Clean Function

```python
# In src/cleanepi/core/clean_data.py
if config.my_function:
    current_data, op_result = _my_cleaning_function(
        current_data, config.my_function
    )
    report.add_operation(op_result)
```

### 5. Write Tests

```python
# tests/test_my_function.py
def test_my_cleaning_function():
    """Test my cleaning function."""
    df = pd.DataFrame({'col': [1, 2, 3]})
    result = my_cleaning_function(df, param1="test")
    assert result.shape == df.shape
```

## Code Style Guidelines

### 1. **Type Hints**
- Use type hints for all function parameters and returns
- Use `Optional[T]` for optional parameters
- Use `Union[T, U]` sparingly, prefer specific types

### 2. **Documentation**
- Follow NumPy docstring format
- Include examples for public functions
- Document all parameters and return values

### 3. **Error Handling**
- Use specific exception types
- Provide helpful error messages
- Log errors appropriately

### 4. **Testing**
- Write tests for all new functionality
- Use descriptive test names
- Test both success and failure cases
- Aim for >90% code coverage

## Performance Considerations

### 1. **Memory Efficiency**
- Use `data.copy()` only when necessary
- Process data in chunks for large datasets
- Monitor memory usage with configurable limits

### 2. **Computation Efficiency**
- Vectorize operations using pandas/numpy
- Avoid Python loops over large datasets
- Use appropriate data types (int32 vs int64, categorical, etc.)

### 3. **I/O Efficiency**
- Use efficient file formats (parquet, feather)
- Implement streaming for large files
- Provide async versions for web applications

## Security Guidelines

### 1. **Input Validation**
- Validate all user inputs using Pydantic
- Sanitize file paths and column names
- Check file sizes and types

### 2. **Safe Operations**
- Avoid `eval()` and `exec()`
- Use safe file operations
- Implement timeouts for long operations

### 3. **Error Information**
- Don't expose internal paths in error messages
- Log security events appropriately
- Handle sensitive data carefully

## Release Process

### 1. **Version Bump**
- Update version in `pyproject.toml`
- Update `__version__` in `__init__.py`
- Update CHANGELOG.md

### 2. **Testing**
- Run full test suite
- Test installation from scratch
- Test examples and documentation

### 3. **Documentation**
- Update README.md
- Update API documentation
- Update usage examples

### 4. **Release**
- Create GitHub release
- Publish to PyPI (when ready)
- Update conda-forge recipe (when ready)

## Getting Help

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Join our community Slack (when available)
- Check the documentation and examples

Thank you for contributing to cleanepi-python! 🎉