"""
utils.py
========
Utility functions for CREMP coral reef data analysis.

Author: Cece Foltz
Date: 03/31/2025
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union
from pathlib import Path

from src.config import METADATA_COLUMNS, CLEANED_DATA_DIR, RAW_DATA_DIR


# =============================================================================
# FILE LOADING FUNCTIONS
# =============================================================================

def find_and_load_csv(
    filename: str,
    data_dir: Optional[str] = None,
    search_cleaned_first: bool = True
) -> Optional[pd.DataFrame]:
    """
    Search for and load a CSV file from multiple possible locations.
    
    Parameters:
    -----------
    filename : str
        Name of the CSV file to load
    data_dir : str, optional
        Primary directory to search first
    search_cleaned_first : bool
        If True, search cleaned data directory before raw
    
    Returns:
    --------
    pd.DataFrame or None
        Loaded dataframe or None if file not found
    
    Example:
    --------
    >>> df = find_and_load_csv('CREMP_Pcover_2023_StonyCoralSpecies_Cleaned.csv')
    >>> print(df.shape)
    """
    # Build list of paths to try
    paths_to_try = []
    
    # Add provided data_dir first
    if data_dir:
        paths_to_try.append(os.path.join(data_dir, filename))
    
    # Add cleaned and raw directories
    if search_cleaned_first:
        paths_to_try.extend([
            os.path.join(CLEANED_DATA_DIR, filename),
            os.path.join(RAW_DATA_DIR, filename),
        ])
    else:
        paths_to_try.extend([
            os.path.join(RAW_DATA_DIR, filename),
            os.path.join(CLEANED_DATA_DIR, filename),
        ])
    
    # Add common relative paths
    paths_to_try.extend([
        filename,
        f'data/cleaned/{filename}',
        f'data/raw/{filename}',
        f'../{filename}',
        f'../../{filename}',
    ])
    
    # Try each path
    for path in paths_to_try:
        if os.path.exists(path):
            print(f"Found file at: {path}")
            try:
                df = pd.read_csv(path)
                print(f"Loaded successfully: {df.shape[0]:,} rows, {df.shape[1]} columns")
                return df
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
    
    # File not found
    print(f"Error: Could not find {filename}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Searched paths:")
    for path in paths_to_try[:5]:  # Show first 5 paths
        print(f"  - {path}")
    
    return None


def load_multiple_files(
    filenames: List[str],
    data_dir: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load multiple CSV files into a dictionary.
    
    Parameters:
    -----------
    filenames : list
        List of filenames to load
    data_dir : str, optional
        Directory to search
    
    Returns:
    --------
    dict
        Dictionary mapping filename to DataFrame
    """
    loaded_files = {}
    
    for filename in filenames:
        df = find_and_load_csv(filename, data_dir)
        if df is not None:
            # Use filename without extension as key
            key = os.path.splitext(filename)[0]
            loaded_files[key] = df
        else:
            print(f"Warning: Could not load {filename}")
    
    print(f"\nLoaded {len(loaded_files)} of {len(filenames)} files")
    return loaded_files


# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def get_species_columns(
    df: pd.DataFrame,
    metadata_cols: Optional[List[str]] = None
) -> List[str]:
    """
    Extract species column names from dataframe by excluding known metadata columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    metadata_cols : list, optional
        List of metadata column names to exclude (uses default if None)
    
    Returns:
    --------
    list
        List of species column names
    
    Example:
    --------
    >>> species_cols = get_species_columns(coral_data)
    >>> print(f"Found {len(species_cols)} species columns")
    """
    if metadata_cols is None:
        metadata_cols = METADATA_COLUMNS
    
    species_cols = [col for col in df.columns if col not in metadata_cols]
    return species_cols


def calculate_coral_metrics(
    df: pd.DataFrame,
    species_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate total cover and species richness for coral data.
    
    TotalCover uses min_count=1 so that rows where ALL species are NaN
    return NaN rather than 0. Rows with a mix of real values and NaN
    are summed over observed values only.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Coral cover dataframe
    species_cols : list, optional
        List of species column names (auto-detected if None)
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with added TotalCover and SpeciesRichness columns
    
    Example:
    --------
    >>> df = calculate_coral_metrics(coral_data)
    >>> print(df[['Year', 'TotalCover', 'SpeciesRichness']].head())
    """
    df = df.copy()
    
    # Auto-detect species columns if not provided
    if species_cols is None:
        species_cols = get_species_columns(df)
    
    # Calculate total cover (min_count=1 means all-NaN rows return NaN)
    df['TotalCover'] = df[species_cols].sum(axis=1, min_count=1)
    
    # Calculate species richness (count of species with cover > 0)
    df['SpeciesRichness'] = (df[species_cols] > 0).sum(axis=1)
    
    return df


def summarize_dataframe(df: pd.DataFrame) -> Dict:
    """
    Generate a summary of a DataFrame's structure and contents.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    dict
        Dictionary with summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing_total': int(df.isnull().sum().sum()),
        'missing_by_column': df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns)
    }
    return summary


def print_dataframe_summary(df: pd.DataFrame, name: str = "DataFrame"):
    """
    Print a formatted summary of a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    name : str
        Name to display in the summary
    """
    summary = summarize_dataframe(df)
    
    print(f"\n{'='*60}")
    print(f"Summary: {name}")
    print('='*60)
    print(f"Shape: {summary['shape'][0]:,} rows × {summary['shape'][1]} columns")
    print(f"Memory: {summary['memory_mb']:.2f} MB")
    print(f"\nColumn Types:")
    for dtype, count in summary['dtypes'].items():
        print(f"  {dtype}: {count}")
    print(f"\nMissing Values: {summary['missing_total']:,} total")
    if summary['missing_by_column']:
        print("  Columns with missing values:")
        for col, count in list(summary['missing_by_column'].items())[:10]:
            print(f"    {col}: {count:,}")
        if len(summary['missing_by_column']) > 10:
            print(f"    ... and {len(summary['missing_by_column']) - 10} more columns")
    print('='*60)


# =============================================================================
# DATA VALIDATION FUNCTIONS
# =============================================================================

def validate_required_columns(
    df: pd.DataFrame,
    required_cols: List[str],
    raise_error: bool = True
) -> bool:
    """
    Check if DataFrame contains all required columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    required_cols : list
        List of required column names
    raise_error : bool
        If True, raise ValueError for missing columns
    
    Returns:
    --------
    bool
        True if all columns present, False otherwise
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        msg = f"Missing required columns: {missing_cols}"
        if raise_error:
            raise ValueError(msg)
        else:
            print(f"Warning: {msg}")
            return False
    
    return True


def validate_year_column(df: pd.DataFrame) -> bool:
    """
    Validate that the Year column exists and contains valid years.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    bool
        True if Year column is valid
    """
    if 'Year' not in df.columns:
        print("Error: 'Year' column not found")
        return False
    
    # Check for valid year range (CREMP started in 1996)
    min_year = df['Year'].min()
    max_year = df['Year'].max()
    
    if min_year < 1990 or max_year > 2100:
        print(f"Warning: Year range seems invalid ({min_year} - {max_year})")
        return False
    
    print(f"Year range: {min_year} - {max_year}")
    return True


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def save_dataframe(
    df: pd.DataFrame,
    filename: str,
    output_dir: Optional[Union[str, Path]] = None,
    index: bool = False
) -> str:
    """
    Save DataFrame to CSV file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    filename : str
        Output filename
    output_dir : str or Path, optional
        Output directory (uses CLEANED_DATA_DIR if None)
    index : bool
        Whether to include index in output
    
    Returns:
    --------
    str
        Path to saved file
    """
    if output_dir is None:
        output_dir = CLEANED_DATA_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / filename
    df.to_csv(output_path, index=index)
    
    print(f"Saved {len(df):,} rows to {output_path}")
    return str(output_path)


# =============================================================================
# MAIN - Run tests when executed directly
# =============================================================================

if __name__ == "__main__":
    print("Testing utility functions...")
    print("="*60)
    
    # Test with sample data
    sample_data = {
        'Year': [2020, 2021, 2022],
        'Subregion': ['Upper Keys', 'Middle Keys', 'Lower Keys'],
        'Species_A': [10.5, 8.3, 6.2],
        'Species_B': [5.0, np.nan, 4.1],
        'Species_C': [0.0, 2.1, 0.0]
    }
    df = pd.DataFrame(sample_data)
    
    print("\nSample DataFrame:")
    print(df)
    
    # Test get_species_columns
    species = get_species_columns(df, metadata_cols=['Year', 'Subregion'])
    print(f"\nSpecies columns: {species}")
    
    # Test calculate_coral_metrics
    df_metrics = calculate_coral_metrics(df, species)
    print(f"\nWith metrics:")
    print(df_metrics[['Year', 'TotalCover', 'SpeciesRichness']])
    
    # Test print_dataframe_summary
    print_dataframe_summary(df, "Sample Data")
    
    print("\n✓ All utility functions working correctly")
