# cleanepi-python: Python Epidemiological Data Cleaning Package

Always follow these instructions first and fallback to search or bash commands only when encountering unexpected information that does not match the info here.

## Working Effectively

### Bootstrap and Setup Environment
- Set up Python virtual environment:
  - `python -m venv venv`
  - `source venv/bin/activate` (Linux/macOS) or `venv\Scripts\activate` (Windows)
  - `pip install --upgrade pip`
- Install package in development mode:
  - `pip install -e ".[dev]"` -- takes 30-45 seconds. NEVER CANCEL. Set timeout to 90+ seconds.
  - For additional features: `pip install -e ".[dev,web,performance,async]"`
  - **Note**: Installation may fail due to network timeouts in constrained environments. This is normal - retry or use existing environment.
- Install missing dependencies if needed:
  - `pip install chardet` (required for CLI file encoding detection)
  - `pip install openpyxl` (required for Excel file support in tests)
  - `pip install pyarrow` (required for Parquet file support in tests)

### Running Tests
- Run basic functionality tests: `pytest tests/test_basic_functionality.py` -- takes 1 second
- Run all core tests (excluding web): `pytest --ignore=tests/unit/test_web_api.py` -- takes 2-3 seconds. Some tests may fail due to missing optional dependencies (openpyxl, pyarrow, chardet) but core functionality will pass.
- Run specific test modules: `pytest tests/unit/test_[module_name].py -v`
- NEVER CANCEL test runs. Core tests complete quickly (under 5 seconds).

### Code Quality and Formatting
- Format code: `black src/ tests/` -- takes 1-2 seconds. NEVER CANCEL.
- Sort imports: `isort src/ tests/` -- takes less than 1 second
- Check code style: `flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics` -- takes less than 1 second
- Type checking: `mypy src/cleanepi --ignore-missing-imports` -- takes 3-5 seconds. NEVER CANCEL.
- **ALWAYS run `black src/ tests/ && isort src/ tests/` before committing** - the code is not properly formatted in the repository

### CLI Usage
- Test CLI functionality: `cleanepi --help`
- Check version: `cleanepi --version`
- Process sample data: `cleanepi input.csv -o output.csv --standardize-columns --replace-missing --remove-duplicates`
- **IMPORTANT**: CLI requires `chardet` dependency for file encoding detection. Install with `pip install chardet`.

### Examples and Validation
- Run basic example: `python examples/basic_usage.py` -- takes 1 second. NEVER CANCEL.
- Run comprehensive example: `python examples/comprehensive_example.py` -- takes 1-2 seconds. NEVER CANCEL.
- ALWAYS run examples after making changes to verify functionality works end-to-end

## Validation Scenarios

After making changes, ALWAYS test these complete user scenarios:

### Basic Python API Usage
```python
import pandas as pd
from cleanepi import clean_data, CleaningConfig

# Create test data
df = pd.DataFrame({
    'Study ID': ['PS001', 'PS002', 'PS001'],
    'Age': ['25', 'unknown', '25'],
    'Status': ['active', 'active', 'active']
})

# Clean data with basic config
config = CleaningConfig(standardize_column_names=True)
cleaned_df, report = clean_data(df, config)

# Verify: column names standardized, data cleaned
assert 'study_id' in cleaned_df.columns
print(report.summary())
```

### CLI Workflow
```bash
# Create sample CSV
echo "ID,Name,Age
1,John,-99
2,Jane,25" > test.csv

# Clean with CLI
cleanepi test.csv -o clean.csv --standardize-columns --replace-missing --na-strings="-99"

# Verify output file created and data cleaned
cat clean.csv
```

### Code Quality Validation
```bash
# Format and check code
black src/ tests/
isort src/ tests/
flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
pytest tests/test_basic_functionality.py
```

## Common Tasks

### Repository Structure
```
cleanepi_py/
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
│   │   └── [other cleaning modules]
│   ├── utils/             # Utility functions
│   ├── cli.py             # Command-line interface
│   └── web/               # Web API (optional)
├── tests/                 # Test suite
│   ├── test_basic_functionality.py  # Basic integration tests
│   └── unit/              # Unit tests by module
├── examples/              # Usage examples
├── pyproject.toml         # Build configuration
└── requirements.txt       # Dependencies
```

### Key Project Information
- **Python version**: 3.9+ (tested on 3.12)
- **Main dependencies**: pandas, numpy, pydantic, loguru
- **CLI entry point**: `cleanepi` command (via `cleanepi.cli:main`)
- **Build system**: setuptools with pyproject.toml
- **Package name**: `cleanepi-python` (PyPI, when published)
- **Import name**: `cleanepi`

### Development Dependencies
The package has several optional dependency groups:
- `dev`: pytest, black, isort, flake8, mypy, pre-commit
- `web`: fastapi, uvicorn, pydantic-settings
- `performance`: dask, pyarrow
- `async`: aiofiles, asyncio-compat

### Troubleshooting Common Issues

#### Import/Module Errors
- **`ModuleNotFoundError: No module named 'chardet'`**: Install with `pip install chardet`
- **`ModuleNotFoundError: No module named 'openpyxl'`**: Install with `pip install openpyxl` for Excel support
- **`ModuleNotFoundError: No module named 'pyarrow'`**: Install with `pip install pyarrow` for Parquet support
- **FastAPI/web errors**: Install web dependencies with `pip install -e ".[web]"`

#### Test Failures
- Some tests may fail due to missing optional dependencies (acceptable for core functionality)
- If web API tests fail, install web dependencies or exclude with `--ignore=tests/unit/test_web_api.py`
- Type checking with mypy shows many errors but this is expected in current codebase state

#### Code Formatting
- Code is not properly formatted in repository - ALWAYS run `black src/ tests/ && isort src/ tests/` before committing
- This is normal and expected - format the code as part of your workflow

### Performance Expectations
- Package installation: 30-45 seconds
- Basic tests: 1-3 seconds
- Code formatting: 1-2 seconds
- Type checking: 3-5 seconds
- Examples execution: 1-2 seconds
- Full test suite (excluding optional deps): 2-5 seconds

### Known Limitations
- Web API functionality requires additional dependencies (`pip install -e ".[web]"`)
- Some file format support (Excel, Parquet) requires additional dependencies
- Type checking shows many errors due to incomplete type annotations
- Code formatting is needed throughout the codebase

## CRITICAL Reminders
- **NEVER CANCEL** any command that takes less than 60 seconds
- **ALWAYS** format code with black and isort before committing
- **ALWAYS** test basic functionality with `python examples/basic_usage.py` after changes
- **ALWAYS** install chardet dependency when using CLI: `pip install chardet`
- Set timeouts of 90+ seconds for installation commands, 30+ seconds for any other commands
- The codebase requires formatting - this is normal, not an error condition