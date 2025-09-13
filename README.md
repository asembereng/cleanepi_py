# cleanepi: Clean and standardize epidemiological data in Python

**cleanepi** is a Python package designed for cleaning, curating, and standardizing epidemiological data. It streamlines various data cleaning tasks that are typically expected when working with datasets in epidemiology.

This package is a Python conversion of the R package [cleanepi](https://github.com/epiverse-trace/cleanepi) developed by the [Epiverse-TRACE](https://data.org/initiatives/epiverse/) team.

## Key Features

**cleanepi** provides the following key functionalities:

1. **Removing irregularities**: Remove duplicated and empty rows and columns, as well as columns with constant values.

2. **Handling missing values**: Replace missing values with the standard `NaN` format, ensuring consistency and ease of analysis.

3. **Ensuring data integrity**: Ensure the uniqueness of uniquely identified columns, maintaining data integrity and preventing duplicates.

4. **Date conversion**: Convert character columns to Date format under specific conditions, enhancing data uniformity and facilitating temporal analysis. Also offers conversion of numeric values written in letters into numbers.

5. **Standardizing entries**: Standardize column entries into specified formats, promoting consistency across the dataset.

6. **Time span calculation**: Calculate the time span between two Date elements, providing valuable demographic insights for epidemiological analysis.

**cleanepi** operates on pandas DataFrames and returns the processed data in the same format, ensuring seamless integration into existing workflows. Additionally, it generates a comprehensive report detailing the outcomes of each cleaning task.

## Installation

You can install **cleanepi** using pip:

```bash
pip install cleanepi
```

Or install from source:

```bash
git clone https://github.com/asembereng/cleanepi_py.git
cd cleanepi_py
pip install -e .
```

## Quick Start

The main function in **cleanepi** is `clean_data()`, which internally calls almost all standard data cleaning functions, such as removal of empty and duplicated rows and columns, replacement of missing values, etc. However, each function can also be called independently to perform a specific task.

```python
import pandas as pd
import numpy as np
from datetime import date
from cleanepi import clean_data, print_report

# Create sample data
data = pd.DataFrame({
    'study_id': ['PS001P2', 'PS002P2', 'PS004P2-1', 'PS003P2', 'P0005P2', 'PS006P2'],
    'event_name': ['day 0', 'day 0', 'day 0', 'day 0', 'day 0', 'day 0'],
    'country_code': [2, 2, 2, 2, 2, 2],
    'country_name': ['Gambia', 'Gambia', 'Gambia', 'Gambia', 'Gambia', 'Gambia'],
    'date_of_admission': ['01/12/2020', '28/01/2021', '15/02/2021', '11/02/2021', '17/02/2021', '17/02/2021'],
    'date_of_birth': ['06/01/1972', '02/20/1952', '06/15/1961', '11/11/1947', '09/26/2000', '-99'],
    'date_first_pcr_positive_test': ['Dec 01, 2020', 'Jan 01, 2021', 'Feb 11, 2021', 'Feb 01, 2021', 'Feb 16, 2021', 'May 02, 2021'],
    'sex': [1, 1, -99, 1, 2, 2]
})

# Define cleaning parameters
replace_missing_values = {'target_columns': None, 'na_strings': ['-99']}
remove_duplicates = {'target_columns': None}
standardize_dates = {
    'target_columns': None,
    'error_tolerance': 0.4,
    'format': None,
    'timeframe': [date(1973, 5, 29), date(2023, 5, 29)]
}
standardize_subject_ids = {
    'target_columns': ['study_id'],
    'prefix': 'PS',
    'suffix': 'P2',
    'range': (1, 100),
    'nchar': 7
}
remove_constants = {'cutoff': 1}
to_numeric = {'target_columns': ['sex'], 'lang': 'en'}

# Perform the data cleaning
cleaned_data = clean_data(
    data=data,
    replace_missing_values=replace_missing_values,
    remove_duplicates=remove_duplicates,
    standardize_dates=standardize_dates,
    standardize_subject_ids=standardize_subject_ids,
    remove_constants=remove_constants,
    to_numeric=to_numeric
)

# Display the cleaning report
print_report(cleaned_data)
```

## Individual Functions

You can also use individual cleaning functions:

```python
from cleanepi import (
    replace_missing_values, 
    remove_duplicates, 
    standardize_date,
    remove_constants,
    convert_to_numeric
)

# Replace missing values
data_clean = replace_missing_values(data, na_strings=['-99', 'missing'])

# Remove duplicates
data_clean = remove_duplicates(data_clean)

# Standardize dates
data_clean = standardize_date(data_clean, target_columns=['date_column'])

# Remove constant columns
data_clean = remove_constants(data_clean, cutoff=1.0)

# Convert to numeric
data_clean = convert_to_numeric(data_clean, target_columns=['age'], lang='en')
```

## Requirements

- Python >= 3.8
- pandas >= 1.3.0
- numpy >= 1.20.0
- python-dateutil >= 2.8.0

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome via [pull requests](https://github.com/asembereng/cleanepi_py/pulls).

## Citation

If you use this package in your research, please cite:

```
cleanepi Python Package
Converted from R package: Mané K, Degoot A, Ahadzie B, Mohammed N, Bah B (2025).
cleanepi: Clean and Standardize Epidemiological Data.
```

## Acknowledgments

This Python package is a conversion of the R package [cleanepi](https://github.com/epiverse-trace/cleanepi) developed by the [Epiverse-TRACE](https://data.org/initiatives/epiverse/) team at the [Medical Research Council The Gambia unit at the London School of Hygiene and Tropical Medicine](https://www.lshtm.ac.uk/research/units/mrc-gambia).