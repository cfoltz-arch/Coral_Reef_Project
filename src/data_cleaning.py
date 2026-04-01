"""
data_cleaning.py
================
Data cleaning pipeline for CREMP coral reef monitoring data.

This script applies the following cleaning decisions:
- Duplicates: Removed from all files
- Missing species values: Left as NaN (reflects unobserved/unsurveyed taxa,
  not confirmed absence — imputing 0 would distort means)
- IQR outlier removal: Applied ONLY to unbounded physical measurements
  (Diameter_cm, Height_cm). NOT applied to percent cover, counts, density,
  LTA, or condition data — these are zero-inflated ecological distributions
  where IQR would flag genuine presence records as outliers
- Negative LTA values: Removed (physically impossible)
- TotalCover: Calculated with min_count=1 so all-NaN rows return NaN, not 0

Author: Cece Foltz
Date: 03/31/2025
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path

from src.config import (
    RAW_DATA_DIR,
    CLEANED_DATA_DIR,
    REPORTS_DIR,
    IQR_OUTLIER_COLUMNS,
    NON_NEGATIVE_COLUMNS,
    METADATA_COLUMNS,
    IQR_MULTIPLIER,
    CHUNK_SIZE,
    ensure_directories
)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CleaningConfig:
    """Configuration for data cleaning pipeline."""
    
    # Directories
    raw_data_dir: Path = RAW_DATA_DIR
    cleaned_data_dir: Path = CLEANED_DATA_DIR
    reports_dir: Path = REPORTS_DIR
    
    # Columns that should have IQR outlier removal applied
    iqr_outlier_columns: List[str] = field(
        default_factory=lambda: IQR_OUTLIER_COLUMNS.copy()
    )
    
    # Columns where negative values are physically impossible
    non_negative_columns: List[str] = field(
        default_factory=lambda: NON_NEGATIVE_COLUMNS.copy()
    )
    
    # Metadata columns (not species data)
    metadata_columns: List[str] = field(
        default_factory=lambda: METADATA_COLUMNS.copy()
    )
    
    # IQR multiplier for outlier detection
    iqr_multiplier: float = IQR_MULTIPLIER


@dataclass
class CleaningReport:
    """Report of cleaning operations performed on a file."""
    filename: str
    original_rows: int
    cleaned_rows: int
    duplicates_removed: int
    outliers_removed: Dict[str, int] = field(default_factory=dict)
    negative_values_removed: Dict[str, int] = field(default_factory=dict)
    missing_values: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# MAIN CLEANER CLASS
# =============================================================================

class CREMPDataCleaner:
    """
    Data cleaner for CREMP coral reef monitoring datasets.
    
    Applies ecologically-appropriate cleaning rules that respect
    the zero-inflated nature of species abundance data.
    
    Attributes:
        config: CleaningConfig object with cleaning parameters
        reports: List of CleaningReport objects for each processed file
    
    Example:
        >>> cleaner = CREMPDataCleaner()
        >>> cleaner.clean_all_files()
        >>> cleaner.print_summary()
    """
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        """
        Initialize the cleaner with configuration.
        
        Args:
            config: CleaningConfig object (uses defaults if None)
        """
        self.config = config or CleaningConfig()
        self.reports: List[CleaningReport] = []
    
    def clean_file(
        self,
        filepath: str,
        output_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        """
        Clean a single CREMP data file.
        
        Args:
            filepath: Path to the raw CSV file
            output_path: Path to save cleaned file (optional)
        
        Returns:
            Tuple of (cleaned DataFrame, CleaningReport)
        """
        filename = os.path.basename(filepath)
        print(f"\n{'='*60}")
        print(f"Cleaning: {filename}")
        print('='*60)
        
        # Load data
        df = pd.read_csv(filepath)
        original_rows = len(df)
        
        # Initialize report
        report = CleaningReport(
            filename=filename,
            original_rows=original_rows,
            cleaned_rows=original_rows,
            duplicates_removed=0
        )
        
        # Step 1: Remove duplicates
        df, dups_removed = self._remove_duplicates(df)
        report.duplicates_removed = dups_removed
        
        # Step 2: Remove negative values from applicable columns
        df, neg_removed = self._remove_negative_values(df)
        report.negative_values_removed = neg_removed
        
        # Step 3: Apply IQR outlier removal ONLY to physical measurement columns
        df, outliers_removed = self._remove_iqr_outliers(df)
        report.outliers_removed = outliers_removed
        
        # Step 4: Document remaining missing values (NOT imputed)
        missing = self._document_missing_values(df)
        report.missing_values = missing
        
        # Update final row count
        report.cleaned_rows = len(df)
        
        # Save cleaned file
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"Saved cleaned file to: {output_path}")
        
        # Store report
        self.reports.append(report)
        
        # Print summary
        self._print_file_report(report)
        
        return df, report
    
    def _remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Remove duplicate rows from DataFrame."""
        original_len = len(df)
        df = df.drop_duplicates()
        duplicates_removed = original_len - len(df)
        
        if duplicates_removed > 0:
            print(f"  ✓ Removed {duplicates_removed} duplicate rows")
        else:
            print(f"  ✓ No duplicates found")
        
        return df, duplicates_removed
    
    def _remove_negative_values(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Remove rows with negative values in columns that cannot be negative.
        
        Only applies to columns like LTA, Diameter, Height where negative
        values are physically impossible.
        """
        removed = {}
        
        for col in self.config.non_negative_columns:
            if col in df.columns:
                neg_mask = df[col] < 0
                neg_count = neg_mask.sum()
                
                if neg_count > 0:
                    df = df[~neg_mask].copy()
                    removed[col] = int(neg_count)
                    print(f"  ✓ Removed {neg_count} negative values from '{col}'")
        
        if not removed:
            print(f"  ✓ No negative values in constrained columns")
        
        return df, removed
    
    def _remove_iqr_outliers(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Remove outliers using IQR method ONLY for physical measurement columns.
        
        IMPORTANT: This is NOT applied to:
        - Percent cover data (zero-inflated, bounded 0-100)
        - Count data (zero-inflated integers)
        - Density data (zero-inflated continuous)
        - LTA data (living tissue area)
        - Condition counts
        """
        removed = {}
        numeric_types = ['float64', 'int64', 'float32', 'int32']
        
        for col in self.config.iqr_outlier_columns:
            if col in df.columns and df[col].dtype in numeric_types:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - self.config.iqr_multiplier * iqr
                upper_bound = q3 + self.config.iqr_multiplier * iqr
                
                outlier_mask = (
                    df[col].notna() &
                    ((df[col] < lower_bound) | (df[col] > upper_bound))
                )
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    df = df[~outlier_mask].copy()
                    removed[col] = int(outlier_count)
                    print(f"  ✓ Removed {outlier_count} IQR outliers from '{col}' "
                          f"(bounds: {lower_bound:.2f} - {upper_bound:.2f})")
        
        if not removed:
            print(f"  ✓ No IQR outlier removal applied (no applicable columns)")
        
        return df, removed
    
    def _document_missing_values(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Document missing values in each column.
        
        Missing values in species columns are LEFT AS NaN intentionally.
        """
        missing = df.isnull().sum()
        missing_dict = missing[missing > 0].to_dict()
        
        if missing_dict:
            total_missing = sum(missing_dict.values())
            print(f"  ℹ Documented {total_missing} missing values across "
                  f"{len(missing_dict)} columns (left as NaN — not imputed)")
        else:
            print(f"  ✓ No missing values")
        
        return {k: int(v) for k, v in missing_dict.items()}
    
    def _print_file_report(self, report: CleaningReport):
        """Print summary report for a cleaned file."""
        print(f"\n  Summary for {report.filename}:")
        print(f"    Original rows: {report.original_rows:,}")
        print(f"    Cleaned rows:  {report.cleaned_rows:,}")
        print(f"    Rows removed:  {report.original_rows - report.cleaned_rows:,}")
        print(f"    Duplicates:    {report.duplicates_removed:,}")
    
    def clean_all_files(self) -> List[CleaningReport]:
        """
        Clean all CREMP data files in the raw data directory.
        
        Returns:
            List of CleaningReport objects
        """
        print("\n" + "="*70)
        print("CREMP DATA CLEANING PIPELINE")
        print("="*70)
        print(f"\nRaw data directory:     {self.config.raw_data_dir}")
        print(f"Cleaned data directory: {self.config.cleaned_data_dir}")
        
        # Ensure output directory exists
        os.makedirs(self.config.cleaned_data_dir, exist_ok=True)
        
        # Get list of CSV files in raw directory
        if not os.path.exists(self.config.raw_data_dir):
            print(f"\nError: Raw data directory not found: {self.config.raw_data_dir}")
            return []
        
        raw_files = [f for f in os.listdir(self.config.raw_data_dir)
                     if f.endswith('.csv')]
        
        if not raw_files:
            print(f"\nNo CSV files found in {self.config.raw_data_dir}")
            return []
        
        print(f"\nFound {len(raw_files)} files to clean")
        
        # Process each file
        for filename in raw_files:
            input_path = os.path.join(self.config.raw_data_dir, filename)
            
            # Generate output filename
            name, ext = os.path.splitext(filename)
            if '_Cleaned' not in name:
                output_filename = f"{name}_Cleaned{ext}"
            else:
                output_filename = filename
            
            output_path = os.path.join(self.config.cleaned_data_dir, output_filename)
            
            # Clean the file
            try:
                self.clean_file(input_path, output_path)
            except Exception as e:
                print(f"\n  ❌ Error cleaning {filename}: {e}")
                self.reports.append(CleaningReport(
                    filename=filename,
                    original_rows=0,
                    cleaned_rows=0,
                    duplicates_removed=0,
                    warnings=[str(e)]
                ))
        
        return self.reports
    
    def print_summary(self):
        """Print overall summary of all cleaning operations."""
        print("\n" + "="*70)
        print("CLEANING PIPELINE SUMMARY")
        print("="*70)
        
        total_original = sum(r.original_rows for r in self.reports)
        total_cleaned = sum(r.cleaned_rows for r in self.reports)
        total_duplicates = sum(r.duplicates_removed for r in self.reports)
        total_removed = total_original - total_cleaned
        
        print(f"\nFiles processed:     {len(self.reports)}")
        print(f"Total original rows: {total_original:,}")
        print(f"Total cleaned rows:  {total_cleaned:,}")
        print(f"Total rows removed:  {total_removed:,}")
        print(f"  - Duplicates:      {total_duplicates:,}")
        
        total_outliers = sum(
            sum(r.outliers_removed.values())
            for r in self.reports
        )
        total_negatives = sum(
            sum(r.negative_values_removed.values())
            for r in self.reports
        )
        
        print(f"  - IQR outliers:    {total_outliers:,}")
        print(f"  - Negative values: {total_negatives:,}")
        
        # Warnings
        all_warnings = [w for r in self.reports for w in r.warnings]
        if all_warnings:
            print(f"\nWarnings ({len(all_warnings)}):")
            for w in all_warnings:
                print(f"  ⚠️ {w}")
        
        print("\n" + "="*70)
        print("Cleaning decisions applied:")
        print("  • Missing species values: Left as NaN (not imputed)")
        print("  • IQR outlier removal: Physical measurements only")
        print("  • Negative values: Removed from constrained columns")
        print("  • Duplicates: Removed from all files")
        print("="*70)
    
    def generate_report_csv(
        self,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a CSV report of all cleaning operations.
        
        Args:
            output_path: Path to save the report
        
        Returns:
            Path to saved report file
        """
        if output_path is None:
            output_path = os.path.join(
                self.config.reports_dir,
                'cleaning_report.csv'
            )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        report_data = []
        for r in self.reports:
            report_data.append({
                'filename': r.filename,
                'original_rows': r.original_rows,
                'cleaned_rows': r.cleaned_rows,
                'rows_removed': r.original_rows - r.cleaned_rows,
                'duplicates_removed': r.duplicates_removed,
                'outliers_removed': sum(r.outliers_removed.values()),
                'negative_values_removed': sum(r.negative_values_removed.values()),
                'missing_value_columns': len(r.missing_values),
                'total_missing_values': sum(r.missing_values.values())
            })
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(output_path, index=False)
        print(f"\nCleaning report saved to: {output_path}")
        
        return output_path


# =============================================================================
# SPECIALIZED CLEANING FUNCTIONS
# =============================================================================

def clean_temperature_data(
    filepath: str,
    output_path: str,
    chunk_size: int = CHUNK_SIZE
):
    """
    Clean large temperature data file using chunked processing.
    
    Args:
        filepath: Path to raw temperature file
        output_path: Path to save cleaned file
        chunk_size: Number of rows per chunk
    """
    print(f"\nCleaning temperature data (chunked processing)...")
    print(f"  Input:  {filepath}")
    print(f"  Output: {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    first_chunk = True
    total_rows = 0
    rows_removed = 0
    
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        original_len = len(chunk)
        
        # Remove duplicates within chunk
        chunk = chunk.drop_duplicates()
        
        # Remove invalid temperatures
        if 'TempC' in chunk.columns:
            valid_mask = (chunk['TempC'] >= 0) & (chunk['TempC'] <= 50)
            chunk = chunk[valid_mask]
        
        rows_removed += original_len - len(chunk)
        total_rows += len(chunk)
        
        # Write chunk to output
        chunk.to_csv(
            output_path,
            mode='w' if first_chunk else 'a',
            header=first_chunk,
            index=False
        )
        first_chunk = False
        
        # Progress indicator
        print(f"  Processed {total_rows:,} rows...", end='\r')
    
    print(f"\n  Total rows processed: {total_rows:,}")
    print(f"  Rows removed: {rows_removed:,}")
    print(f"  Saved to: {output_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run the complete data cleaning pipeline."""
    
    # Ensure directories exist
    ensure_directories()
    
    # Initialize cleaner with default configuration
    config = CleaningConfig()
    cleaner = CREMPDataCleaner(config)
    
    # Clean all files
    cleaner.clean_all_files()
    
    # Print summary
    cleaner.print_summary()
    
    # Generate CSV report
    cleaner.generate_report_csv()
    
    print("\n✅ Data cleaning pipeline complete!")


if __name__ == "__main__":
    main()
