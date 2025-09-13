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

# Advanced date standardization with intelligent parsing
date_config = DateConfig(
    target_columns=["admission_date", "test_date", "birth_date"],
    formats=["dd/mm/yyyy", "yyyy-mm-dd", "%d %b %Y"],
    timeframe=("1900-01-01", "2024-12-31"),
    error_tolerance=0.1
)

cleaned_data = standardize_dates(data, date_config)

# Auto-detect date columns
auto_cleaned = standardize_dates(data)  # Automatically finds and converts date columns
```

### Subject ID Validation

```python
from cleanepi import check_subject_ids, SubjectIDConfig

# Validate subject IDs with pattern matching
id_config = SubjectIDConfig(
    target_columns=["patient_id", "study_id"],
    prefix="P",
    nchar=4,
    range=(1, 9999),
    pattern=r"^P\d{3}$"  # Custom regex pattern
)

validated_data = check_subject_ids(data, id_config)
# Adds validation columns: patient_id_valid, patient_id_issues
```

### Intelligent Numeric Conversion

```python
from cleanepi import convert_to_numeric, NumericConfig

# Convert text to numbers with multi-language support
numeric_config = NumericConfig(
    target_columns=["age", "income", "percentage"],
    lang="en",  # Supports "en", "es", "fr"
    errors="coerce"
)

# Handles: "twenty-five", "$50,000", "85%", "10-15" (ranges)
converted_data = convert_to_numeric(data, numeric_config)
```

### Dictionary-Based Cleaning

```python
from cleanepi import clean_using_dictionary, create_mapping_dictionary

# Define value mappings for standardization
dictionary = {
    "sex": {"1": "male", "2": "female", "m": "male", "f": "female"},
    "status": {"0": "negative", "1": "positive", "pos": "positive", "neg": "negative"},
    "result": {"abnormal": "abnormal", "normal": "normal", "abn": "abnormal"}
}

cleaned_data = clean_using_dictionary(
    data, 
    dictionary,
    case_sensitive=False,
    exact_match=False,  # Allow partial matching
    default_action="keep"  # Keep unmapped values
)

# Generate template dictionary from existing data
template = create_mapping_dictionary(data, ["status", "result"])
```

### Date Sequence Validation

```python
from cleanepi import check_date_sequence, generate_date_sequence_report

# Validate chronological order of dates
sequence_data = check_date_sequence(
    data,
    target_columns=["birth_date", "admission_date", "discharge_date"],
    tolerance_days=1,  # Allow 1-day tolerance
    allow_equal=True,
    subject_id_column="patient_id"  # Validate per patient
)

# Generate comprehensive validation report
report = generate_date_sequence_report(sequence_data, ["birth_date", "admission_date", "discharge_date"])
print(f"Valid sequences: {report['summary']['valid_percentage']:.1f}%")
```

## Comprehensive Example

Here's a complete example showcasing all Phase 2 features working together:

```python
import pandas as pd
from cleanepi import clean_data, CleaningConfig
from cleanepi.core.config import DateConfig, SubjectIDConfig, NumericConfig

# Sample epidemiological data with common data quality issues
data = pd.DataFrame({
    'patient_id': ['P001', 'P002', 'P003', 'P004', 'p005'],
    'birth_date': ['1990-01-15', '15/02/1985', '1995-07-10', '20 Mar 1992', 'unknown'],
    'admission_date': ['2023-01-10', '2023/02/15', '2023-01-05', '2023-03-20', '2023-01-01'],
    'discharge_date': ['2023-01-20', '2023/02/10', '2023-01-15', '2023-03-25', '2023-01-05'],
    'age_text': ['25', 'thirty-five', '28', 'twenty-eight', 'unknown'],
    'test_result': ['pos', 'neg', 'positive', '1', '0'],
    'score_pct': ['85%', '90.5%', 'seventy-five percent', '80%', 'N/A'],
    'income': ['$50,000', '75000', 'sixty thousand', '$45,500', 'missing']
})

# Comprehensive cleaning configuration
config = CleaningConfig(
    # Phase 1 features (already implemented)
    standardize_column_names=True,
    replace_missing_values=True,
    remove_duplicates=True,
    remove_constants=True,
    
    # Phase 2 features (newly implemented)
    standardize_dates=DateConfig(
        target_columns=['birth_date', 'admission_date', 'discharge_date'],
        timeframe=('1900-01-01', '2024-12-31'),
        error_tolerance=0.2
    ),
    
    standardize_subject_ids=SubjectIDConfig(
        target_columns=['patient_id'],
        prefix='P',
        nchar=4,
        range=(1, 9999)
    ),
    
    to_numeric=NumericConfig(
        target_columns=['age_text', 'score_pct', 'income'],
        lang='en',
        errors='coerce'
    ),
    
    dictionary={
        'test_result': {
            'pos': 'positive',
            'neg': 'negative', 
            '1': 'positive',
            '0': 'negative'
        }
    },
    
    check_date_sequence=['birth_date', 'admission_date', 'discharge_date']
)

# Perform comprehensive cleaning
cleaned_data, report = clean_data(data, config)

# Display results
print("=== CLEANING RESULTS ===")
print(f"Original shape: {data.shape}")
print(f"Cleaned shape: {cleaned_data.shape}")
print(f"Total operations: {len(report.operations)}")
print(f"Processing time: {report.duration:.2f}s")

print("\n=== VALIDATION RESULTS ===")
# Subject ID validation results
if 'patient_id_valid' in cleaned_data.columns:
    valid_ids = cleaned_data['patient_id_valid'].sum()
    total_ids = len(cleaned_data)
    print(f"Valid patient IDs: {valid_ids}/{total_ids}")

# Date sequence validation results  
if 'date_sequence_valid' in cleaned_data.columns:
    valid_sequences = cleaned_data['date_sequence_valid'].sum()
    print(f"Valid date sequences: {valid_sequences}/{total_ids}")

# Display cleaning report summary
print("\n=== DETAILED REPORT ===")
print(report.summary())
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