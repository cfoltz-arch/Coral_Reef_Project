"""
Coral Reef Monitoring and Analysis Package
==========================================

A toolkit for analyzing CREMP coral reef monitoring data.

Author: Cece Foltz
Date: 03/31/2025

Modules:
    - analysis: Long-term trend analysis and predictions
    - data_cleaning: Ecologically-appropriate data cleaning
    - config: Configuration constants
    - utils: Utility functions
"""

__version__ = "1.0.0"
__author__ = "Cece Foltz"

# Import key functions for easy access
from src.config import (
    METADATA_COLUMNS,
    RAW_DATA_DIR,
    CLEANED_DATA_DIR,
    CHUNK_SIZE,
    get_raw_filepath,
    get_cleaned_filepath,
    ensure_directories
)

from src.utils import (
    find_and_load_csv,
    get_species_columns,
    calculate_coral_metrics
)

__all__ = [
    'METADATA_COLUMNS',
    'RAW_DATA_DIR', 
    'CLEANED_DATA_DIR',
    'CHUNK_SIZE',
    'get_raw_filepath',
    'get_cleaned_filepath',
    'ensure_directories',
    'find_and_load_csv',
    'get_species_columns',
    'calculate_coral_metrics'
]
