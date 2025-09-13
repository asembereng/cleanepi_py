# cleanepi-python: Data Cleaning and Standardization Package

cleanepi-python is a Python package for cleaning, curating, and standardizing epidemiological data. It provides comprehensive data cleaning operations including column standardization, missing value handling, duplicate removal, and data validation.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap, build, and test the repository:

```bash
# Install development dependencies (takes ~30 seconds)
pip install -e ".[dev]"

# Install optional dependencies for full functionality
pip install openpyxl pyarrow fastapi uvicorn httpx python-multipart

# Run basic tests (takes ~0.7 seconds) 
pytest tests/test_basic_functionality.py -v

# Run unit tests without web API (takes ~2.5 seconds)
pytest tests/unit/ -v --ignore=tests/unit/test_web_api.py

# Run full test suite with all dependencies (takes ~3 seconds, 1 test failure expected)
pytest tests/ -v
```

**NEVER CANCEL**: Test runs complete in under 5 seconds. Always wait for completion.

### Code quality and linting:

```bash
# Check code formatting (takes ~2 seconds) - NOTE: Will show many files need reformatting
black --check src/ tests/

# Apply code formatting
black src/ tests/

# Check import sorting (takes ~0.2 seconds) - NOTE: Will show many import sorting issues
isort --check-only src/ tests/

# Apply import sorting
isort src/ tests/

# Run linting (takes ~0.8 seconds) - NOTE: Will show many violations initially
flake8 src/ tests/

# Type checking (takes ~6 seconds) - NOTE: Will show 80+ errors initially due to missing pandas stubs
mypy src/
```

**NEVER CANCEL**: All linting tools complete in under 10 seconds. Always wait for completion.

### Run the CLI tool:

```bash
# Test CLI functionality (takes ~0.5 seconds)
cleanepi --help

# Example data cleaning with CLI
cleanepi input.csv -o output.csv --standardize-columns --replace-missing --na-strings "-99" --remove-duplicates --remove-constants
```

### Run coverage analysis:

```bash
# Run tests with coverage (takes ~1.5 seconds)
pytest --cov=src/cleanepi --cov-report=term-missing tests/test_basic_functionality.py
```

### Run example scripts:

```bash
# Test basic functionality example (takes ~0.6 seconds)
python examples/basic_usage.py
```

**NEVER CANCEL**: All operations complete quickly. Build and test times are under 30 seconds total.

## Validation

- **ALWAYS run basic tests first** using `pytest tests/test_basic_functionality.py -v` to verify core functionality
- **ALWAYS run the CLI tool** with `cleanepi --help` and test with sample data to ensure it works
- **ALWAYS test the example script** with `python examples/basic_usage.py` to validate end-to-end functionality
- **ALWAYS run linting tools** before committing: `black src/ tests/`, `isort src/ tests/`, `flake8 src/ tests/`
- The repository has code formatting issues that need to be addressed - 33 files need black formatting and many have flake8 violations
- Some tests may fail due to missing optional dependencies or pandas data construction issues, but core functionality works

## Build System and Dependencies

- **Package manager**: pip with pyproject.toml configuration
- **Python version**: 3.9+ (tested with 3.12.3)
- **Core dependencies**: pandas, numpy, pydantic, loguru, python-dateutil
- **Optional dependencies**:
  - `[dev]`: pytest, black, isort, flake8, mypy, pre-commit
  - `[web]`: fastapi, uvicorn, pydantic-settings (requires httpx, python-multipart for tests)
  - `[performance]`: dask, pyarrow  
  - `[async]`: aiofiles
- **Installation command**: `pip install -e ".[dev]"` for development mode

## Repository Structure

### Key directories and files:
```
.
├── src/cleanepi/           # Main package source
│   ├── __init__.py         # Package exports
│   ├── cli.py              # Command-line interface
│   ├── cleaning/           # Data cleaning modules
│   ├── core/               # Core functionality (config, clean_data, report)
│   ├── utils/              # Utility functions (validation, data_scanning)
│   └── web/                # Web API components
├── tests/                  # Test suite
│   ├── test_basic_functionality.py  # Basic integration tests
│   └── unit/               # Unit tests for each module
├── examples/               # Usage examples
├── pyproject.toml          # Package configuration
├── README.md               # Package documentation
├── INSTALL.md              # Installation guide
└── CONTRIBUTING.md         # Development guide
```

### Core modules:
- `src/cleanepi/core/clean_data.py` - Main data cleaning function
- `src/cleanepi/core/config.py` - Configuration models using Pydantic
- `src/cleanepi/core/report.py` - Cleaning operation reporting
- `src/cleanepi/cli.py` - Command-line interface
- `src/cleanepi/cleaning/` - Individual cleaning operations
- `src/cleanepi/utils/validation.py` - Data validation utilities

## Current State and Known Issues

### What works:
- ✅ Package installation and basic functionality
- ✅ Core data cleaning operations (standardize columns, remove duplicates, etc.)
- ✅ CLI tool functionality
- ✅ Basic and unit tests (276 pass, 1 fails)
- ✅ Configuration system with Pydantic models
- ✅ Comprehensive reporting system
- ✅ Example scripts

### Known issues that need fixing:
- ❌ Code formatting: 33 files need black formatting
- ❌ Import sorting: Many files have incorrect import order
- ❌ Linting: Many flake8 violations (line length, whitespace, unused imports)
- ❌ Type checking: 80+ mypy errors (missing pandas stubs, type annotations)
- ❌ One test failure in `test_remove_constants.py` due to pandas array length mismatch

### Dependencies that require manual installation for full testing:
- `openpyxl` - For Excel file support
- `pyarrow` - For Parquet file support  
- `fastapi`, `uvicorn`, `httpx`, `python-multipart` - For web API functionality

## Development Workflow

### Before making changes:
1. **Always run basic tests**: `pytest tests/test_basic_functionality.py -v`
2. **Test CLI functionality**: `cleanepi --help`
3. **Run example script**: `python examples/basic_usage.py`

### After making changes:
1. **Format code**: `black src/ tests/`
2. **Sort imports**: `isort src/ tests/`
3. **Run linting**: `flake8 src/ tests/`
4. **Run tests**: `pytest tests/test_basic_functionality.py -v`
5. **Test CLI**: Test with actual data files
6. **Verify examples still work**: `python examples/basic_usage.py`

### CI/Build validation:
- The repository does not have CI set up yet
- Manual validation steps above substitute for CI checks
- Always run all linting and testing commands before committing

## Common Tasks

### Testing data cleaning functionality:
```bash
# Create test data
echo "Patient ID,Date of Birth,Status,Constant Col
1,1990-01-01,positive,same
2,-99,negative,same
1,1990-01-01,positive,same" > test_data.csv

# Clean with CLI
cleanepi test_data.csv -o cleaned.csv --standardize-columns --replace-missing --na-strings "-99" --remove-duplicates --remove-constants

# Verify output
cat cleaned.csv
```

### Running specific test categories:
```bash
# Basic functionality tests only
pytest tests/test_basic_functionality.py -v

# Unit tests without web API
pytest tests/unit/ -v --ignore=tests/unit/test_web_api.py

# Tests with coverage
pytest --cov=src/cleanepi tests/test_basic_functionality.py
```

### Package development:
```bash
# Install in development mode
pip install -e ".[dev]"

# Check package imports
python -c "from cleanepi import clean_data, CleaningConfig; print('Import successful')"

# Test CLI installation
which cleanepi
cleanepi --version
```

## Performance Expectations

- **Package installation**: ~30 seconds
- **Basic tests**: ~0.7 seconds (11 tests)
- **Unit tests**: ~2.5 seconds (241 tests, some may fail due to dependencies)
- **Full test suite**: ~3 seconds (277 tests)
- **CLI operations**: ~0.5 seconds for typical data cleaning
- **Code formatting (black)**: ~2 seconds
- **Linting (flake8)**: ~0.8 seconds  
- **Type checking (mypy)**: ~6 seconds
- **Import sorting (isort)**: ~0.2 seconds
- **Coverage analysis**: ~1.5 seconds
- **Example script**: ~0.6 seconds

All operations complete quickly - **NEVER CANCEL** any build or test commands.

## Package Usage Examples

### Python API:
```python
import pandas as pd
from cleanepi import clean_data, CleaningConfig
from cleanepi.core.config import MissingValueConfig, DuplicateConfig

# Configure cleaning
config = CleaningConfig(
    standardize_column_names=True,
    replace_missing_values=MissingValueConfig(na_strings=['-99', 'unknown']),
    remove_duplicates=DuplicateConfig(keep='first'),
    remove_constants=True
)

# Clean data
cleaned_df, report = clean_data(df, config)
report.print_summary()
```

### CLI Usage:
```bash
cleanepi input.csv -o output.csv \
  --standardize-columns \
  --replace-missing \
  --na-strings "-99" \
  --remove-duplicates \
  --remove-constants
```

This package provides robust data cleaning capabilities with comprehensive reporting and validation - always test your changes with real data scenarios to ensure functionality works as expected.