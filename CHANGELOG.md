# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial Python implementation of cleanepi R package
- Core data cleaning orchestration with `clean_data()` function
- Column name standardization with multiple naming conventions
- Missing value replacement with 50+ built-in patterns
- Duplicate row removal with flexible criteria
- Constant column removal with configurable thresholds
- Comprehensive configuration system using Pydantic
- Detailed operation reporting and logging
- Security-focused input validation
- Memory usage monitoring and limits
- Basic test suite with 11 passing tests
- Example scripts and documentation

### TODO
- Date standardization and parsing
- Subject ID validation and formatting
- Numeric conversion with language support
- Dictionary-based value cleaning
- Date sequence validation
- Web application components (FastAPI)
- Async processing for large datasets
- Performance optimizations with Dask
- Complete test coverage
- Documentation website

## [0.1.0] - 2024-12-09

### Added
- Initial package structure
- Project configuration with pyproject.toml
- Modern Python packaging setup
- Development environment configuration

---

## Release Notes

### Version 0.1.0 - Initial Release

This is the initial release of cleanepi-python, a Python port of the R package cleanepi for cleaning and standardizing epidemiological data.

**Key Features:**
- Modern Python package structure with src/ layout
- Type-safe configuration using Pydantic
- Comprehensive logging and reporting
- Security-focused design
- Scalable architecture for web applications

**Core Functionality:**
- Column name standardization
- Missing value detection and replacement
- Duplicate removal
- Constant column removal
- Main orchestration function

**Quality Assurance:**
- 11 comprehensive tests
- Input validation and error handling
- Example scripts and documentation
- Security considerations

**Future Roadmap:**
- Complete implementation of remaining R functions
- Web application components
- Performance optimizations
- Async processing support

This release provides a solid foundation for the Python version while maintaining compatibility with the original R package's functionality and design principles.