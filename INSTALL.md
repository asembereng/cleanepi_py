# Installation Guide

This guide provides detailed instructions for installing and setting up cleanepi-python.

## Requirements

- Python 3.9 or higher
- pip (Python package installer)

## Basic Installation

### Option 1: Install from PyPI (Coming Soon)

```bash
pip install cleanepi-python
```

### Option 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/epiverse-trace/cleanepi.git
cd cleanepi/cleanepi_python

# Install in development mode
pip install -e .
```

## Installation with Optional Dependencies

### For Web Applications

```bash
pip install cleanepi-python[web]
```

This includes:
- FastAPI for REST API endpoints
- Uvicorn for ASGI server
- Pydantic-settings for configuration management

### For Performance (Large Datasets)

```bash
pip install cleanepi-python[performance]
```

This includes:
- Dask for distributed computing
- PyArrow for efficient data formats

### For Async Processing

```bash
pip install cleanepi-python[async]
```

This includes:
- aiofiles for async file operations
- Additional async utilities

### For Development

```bash
pip install cleanepi-python[dev]
```

This includes:
- pytest for testing
- black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking
- pre-commit for git hooks

### Complete Installation

```bash
pip install cleanepi-python[web,performance,async,dev]
```

## Virtual Environment Setup (Recommended)

### Using venv (Python 3.3+)

```bash
# Create virtual environment
python -m venv cleanepi_env

# Activate virtual environment
# On Windows:
cleanepi_env\Scripts\activate
# On macOS/Linux:
source cleanepi_env/bin/activate

# Install package
pip install cleanepi-python
```

### Using conda

```bash
# Create conda environment
conda create -n cleanepi_env python=3.11

# Activate environment
conda activate cleanepi_env

# Install package
pip install cleanepi-python
```

### Using poetry (for developers)

```bash
# Install poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Clone repository
git clone https://github.com/epiverse-trace/cleanepi.git
cd cleanepi/cleanepi_python

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

## Verification

After installation, verify that the package is working correctly:

```python
import cleanepi
import pandas as pd

# Create sample data
data = pd.DataFrame({
    'Date of Birth': ['1990-01-01', '-99', '1985-05-15'],
    'Patient ID': [1, 2, 3],
    'Status': ['positive', 'negative', 'unknown']
})

# Test basic functionality
config = cleanepi.CleaningConfig()
cleaned_data, report = cleanepi.clean_data(data, config)

print("Installation successful!")
print(f"Cleaned data shape: {cleaned_data.shape}")
```

## Troubleshooting

### Common Issues

#### 1. Python Version Compatibility

```bash
# Check Python version
python --version

# If you have multiple Python versions
python3.9 -m pip install cleanepi-python
```

#### 2. Permission Issues

```bash
# Install for current user only
pip install --user cleanepi-python

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install cleanepi-python
```

#### 3. Dependency Conflicts

```bash
# Create clean environment
python -m venv fresh_env
source fresh_env/bin/activate
pip install --upgrade pip
pip install cleanepi-python
```

#### 4. Windows-Specific Issues

```bash
# If you encounter SSL errors
pip install --trusted-host pypi.org --trusted-host pypi.python.org cleanepi-python

# If you have path issues
python -m pip install cleanepi-python
```

### Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/epiverse-trace/cleanepi/issues)
2. Create a new issue with:
   - Your operating system and version
   - Python version (`python --version`)
   - Full error message
   - Steps to reproduce the problem

## Development Installation

For contributors and developers:

```bash
# Clone the repository
git clone https://github.com/epiverse-trace/cleanepi.git
cd cleanepi/cleanepi_python

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,web,performance,async]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest
```

## Docker Installation (Coming Soon)

```bash
# Pull Docker image
docker pull cleanepi/cleanepi-python:latest

# Run container
docker run -it cleanepi/cleanepi-python:latest python
```

## System-Specific Instructions

### Ubuntu/Debian

```bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip python3-venv

# Install package
pip3 install cleanepi-python
```

### CentOS/RHEL/Fedora

```bash
# Install Python and pip
sudo dnf install python3 python3-pip

# Install package
pip3 install cleanepi-python
```

### macOS

```bash
# Install using Homebrew (if not already installed)
brew install python

# Install package
pip3 install cleanepi-python
```

### Windows

1. Download Python from [python.org](https://www.python.org/downloads/)
2. Make sure to check "Add Python to PATH" during installation
3. Open Command Prompt or PowerShell
4. Install the package:

```bash
pip install cleanepi-python
```

## Next Steps

After successful installation:

1. Read the [Quick Start Guide](README.md#quick-start)
2. Check out the [examples](examples/)
3. Read the [API documentation](docs/)
4. Join our community discussions