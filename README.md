# cleanepi-python: Clean and Standardize Epidemiological Data

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**cleanepi-python** is a Python package designed for cleaning, curating, and standardizing epidemiological data. This is a Python port of the original R package [cleanepi](https://github.com/epiverse-trace/cleanepi), built with a focus on scalability, reliability, and security for production environments and web applications.

## Key Features

1. **Data Standardization**: 
   - Clean and standardize column names
   - Standardize date formats across multiple input formats
   - Validate and standardize subject IDs

2. **Data Cleaning**: 
   - Remove duplicate rows with flexible criteria
   - Remove empty rows and columns
   - Remove constant columns

3. **Missing Value Handling**: 
   - Replace various missing value representations with standard NaN
   - Support for custom missing value patterns

4. **Data Conversion**: 
   - Convert character columns to numeric
   - Intelligent date parsing and conversion
   - Handle mixed data types

5. **Dictionary-Based Cleaning**: 
   - Replace coded values using data dictionaries
   - Support for complex value mappings

6. **Date Sequence Validation**: 
   - Check chronological order of date events
   - Flag inconsistent date sequences

7. **Comprehensive Reporting**: 
   - Detailed reports of all cleaning operations
   - Tracking of data quality metrics

## Design Principles

- **Scalability**: Built with pandas and numpy for efficient data processing, with optional Dask support for large datasets
- **Reliability**: Comprehensive error handling, logging, and validation
- **Security**: Input validation, safe file operations, protection against injection attacks
- **Web-Ready**: Structured for easy integration into web applications and APIs

## Installation

```bash
# Basic installation
pip install cleanepi-python

# With optional dependencies for web applications
pip install cleanepi-python[web]

# With performance enhancements for large datasets
pip install cleanepi-python[performance]

# Development installation
pip install cleanepi-python[dev]
```

## Quick Start

```python
import pandas as pd
from cleanepi import clean_data, CleaningConfig

# Load your data
data = pd.read_csv("your_epidemiological_data.csv")

# Configure cleaning operations
config = CleaningConfig(
    standardize_column_names=True,
    remove_duplicates=True,
    standardize_dates=True,
    replace_missing_values={
        "na_strings": ["-99", "N/A", "NULL", ""]
    }
)

# Clean the data
cleaned_data, report = clean_data(data, config)

# View the cleaning report
print(report.summary())
```

## Advanced Usage

### Custom Date Parsing

```python
from cleanepi import standardize_dates, DateConfig

date_config = DateConfig(
    target_columns=["admission_date", "test_date"],
    formats=["dd/mm/yyyy", "yyyy-mm-dd"],
    timeframe=("1990-01-01", "2024-12-31"),
    error_tolerance=0.1
)

cleaned_data = standardize_dates(data, date_config)
```

### Dictionary-Based Cleaning

```python
from cleanepi import clean_using_dictionary

# Define value mappings
dictionary = {
    "sex": {"1": "male", "2": "female", "m": "male", "f": "female"},
    "status": {"0": "negative", "1": "positive"}
}

cleaned_data = clean_using_dictionary(data, dictionary)
```

### Async Processing for Large Datasets

```python
import asyncio
from cleanepi.async_processing import async_clean_data

async def process_large_dataset():
    cleaned_data, report = await async_clean_data(
        data_source="large_dataset.parquet",
        config=config,
        chunk_size=10000
    )
    return cleaned_data, report

# Run async processing
result = asyncio.run(process_large_dataset())
```

## Web Application Integration

The package is designed to integrate seamlessly with web frameworks:

```python
from fastapi import FastAPI, UploadFile
from cleanepi.web import create_cleaning_endpoint

app = FastAPI()

# Add data cleaning endpoint
app.include_router(create_cleaning_endpoint(), prefix="/api/v1")
```

## Security Features

- Input validation using Pydantic models
- Safe file operations with size and type restrictions
- Protection against code injection in custom cleaning rules
- Secure handling of temporary files
- Audit logging of all operations

## Performance Considerations

- Efficient memory usage with pandas optimizations
- Optional Dask integration for datasets that don't fit in memory
- Streaming processing for very large files
- Configurable chunk processing
- Progress tracking for long-running operations

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{cleanepi_python,
  title = {cleanepi-python: Clean and Standardize Epidemiological Data},
  author = {Mané, Karim and Degoot, Abdoelnaser and others},
  year = {2024},
  url = {https://github.com/epiverse-trace/cleanepi},
}
```

## Acknowledgments

This Python package is based on the original R package [cleanepi](https://github.com/epiverse-trace/cleanepi) developed by the [Epiverse-TRACE](https://data.org/initiatives/epiverse/) team at the Medical Research Council The Gambia unit at the London School of Hygiene and Tropical Medicine.