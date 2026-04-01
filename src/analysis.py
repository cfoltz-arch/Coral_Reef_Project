"""
analysis.py
===========
Coral Reef Monitoring and Analysis Script

This script analyzes CREMP (Coral Reef Evaluation and Monitoring Project) data
to assess coral reef health trends, species richness, and future projections.

Author: Cece Foltz
Date: 03/31/2026

Cleaning decisions applied (see data_cleaning.py for details):
- Duplicates removed from raw files
- Missing species values left as NaN
- IQR outlier removal applied ONLY to physical measurements
- Negative LTA values removed
- TotalCover calculated with min_count=1
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from typing import Dict, List, Tuple, Optional

from src.config import (
    METADATA_COLUMNS,
    CHUNK_SIZE,
    CLEANED_DATA_DIR,
    RAW_DATA_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    FIG_SIZE_SINGLE,
    FIG_SIZE_DOUBLE,
    FIG_SIZE_TALL,
    ensure_directories
)
from src.utils import (
    find_and_load_csv,
    get_species_columns,
    calculate_coral_metrics,
    print_dataframe_summary
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# List of cleaned data files for analysis
DATA_FILES = [
    'CREMP_OCTO_RawData_2023_Cleaned.csv',
    'CREMP_OCTO_Summaries_2023_Density_Cleaned.csv',
    'CREMP_OCTO_Summaries_2023_MeanHeight_Cleaned.csv',
    'CREMP_Pcover_2023_StonyCoralSpecies_Cleaned.csv',
    'CREMP_Pcover_2023_TaxaGroups_Cleaned.csv',
    'CREMP_SCOR_RawData_2023_Cleaned.csv',
    'CREMP_SCOR_Summaries_2023_ConditionCounts_Cleaned.csv',
    'CREMP_SCOR_Summaries_2023_Counts_Cleaned.csv',
    'CREMP_SCOR_Summaries_2023_Density_Cleaned.csv',
    'CREMP_SCOR_Summaries_2023_LTA_Cleaned.csv',
    'CREMP_Stations_2023_Cleaned.csv',
    'CREMP_Temperatures_2023.csv'
]


# =============================================================================
# AUDIT FUNCTIONS
# =============================================================================

def analyze_missing_and_outliers(filepath: str) -> Optional[Dict]:
    """
    Audit a cleaned CSV file for remaining missing values and outlier flags.
    
    NOTE: Outlier counts here are for informational purposes only.
    IQR is not an appropriate removal criterion for zero-inflated
    species matrices.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    
    Returns:
    --------
    dict or None
        Dictionary containing analysis results
    """
    print("-" * 50)
    print(f"Analyzing file: {os.path.basename(filepath)}")
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    results = {
        'shape': df.shape,
        'missing_values': {},
        'outliers': {},
        'dtypes': df.dtypes.to_dict()
    }
    
    print(f"Shape: {df.shape}")
    print(f"Head:\n{df.head()}")
    
    # Missing values
    missing_summary = df.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]
    if len(missing_cols) > 0:
        print(f"\nMissing values per column (left as NaN):")
        for col, count in missing_cols.items():
            print(f"  {col}: {count}")
        results['missing_values'] = missing_cols.to_dict()
    else:
        print("\nNo missing values.")
    
    # Outlier audit (informational only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print("\nOutlier audit (IQR method — informational only):")
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_count = len(df[
                (df[col].notna()) &
                ((df[col] < lower_bound) | (df[col] > upper_bound))
            ])
            results['outliers'][col] = outlier_count
            if outlier_count > 0:
                print(f"  {col}: {outlier_count} flagged")
    else:
        print("\nNo numeric columns.")
    
    print("=" * 50 + "\n")
    
    return results


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_long_term_trends(
    coral_data: pd.DataFrame,
    species_cols: List[str],
    save_figure: bool = False
) -> Dict:
    """
    Perform long-term trend analysis on coral cover and species richness.
    
    Parameters:
    -----------
    coral_data : pd.DataFrame
        Coral cover data with Year column
    species_cols : list
        List of species column names
    save_figure : bool
        If True, save figure to outputs/figures/
    
    Returns:
    --------
    dict
        Dictionary containing trend analysis results including:
        - yearly_trends: DataFrame with yearly means
        - cover_trend: dict with slope, intercept, r_squared, p_value
        - richness_trend: dict with slope, intercept, r_squared, p_value
    """
    print("\n" + "=" * 60)
    print("LONG-TERM TREND ANALYSIS")
    print("=" * 60)
    
    # Calculate metrics if not present
    if 'TotalCover' not in coral_data.columns:
        coral_data = calculate_coral_metrics(coral_data, species_cols)
    
    # Calculate yearly means (NaN rows excluded automatically)
    yearly_trends = coral_data.groupby('Year').agg({
        'TotalCover': 'mean',
        'SpeciesRichness': 'mean'
    }).reset_index()
    
    # Sort by year for regression
    yearly_trends = yearly_trends.sort_values('Year')
    
    print("\nYearly trends (mean values):")
    print(yearly_trends.to_string(index=False))
    
    # Linear regression for total cover
    slope_cover, intercept_cover, r_cover, p_cover, se_cover = stats.linregress(
        yearly_trends['Year'], yearly_trends['TotalCover']
    )
    
    # Linear regression for species richness
    slope_rich, intercept_rich, r_rich, p_rich, se_rich = stats.linregress(
        yearly_trends['Year'], yearly_trends['SpeciesRichness']
    )
    
    # Print results
    print("\n" + "-" * 40)
    print("TOTAL CORAL COVER TREND:")
    print("-" * 40)
    print(f"  Slope (change per year): {slope_cover:.4f}%")
    print(f"  Intercept:               {intercept_cover:.4f}")
    print(f"  R-squared:               {r_cover**2:.4f}")
    print(f"  P-value:                 {p_cover:.6f}")
    print(f"  Standard error:          {se_cover:.4f}")
    
    if p_cover < 0.05:
        trend_direction = "declining" if slope_cover < 0 else "increasing"
        print(f"  *** Statistically significant {trend_direction} trend (p < 0.05) ***")
    else:
        print(f"  No statistically significant trend (p >= 0.05)")
    
    print("\n" + "-" * 40)
    print("SPECIES RICHNESS TREND:")
    print("-" * 40)
    print(f"  Slope (change per year): {slope_rich:.4f} species")
    print(f"  Intercept:               {intercept_rich:.4f}")
    print(f"  R-squared:               {r_rich**2:.4f}")
    print(f"  P-value:                 {p_rich:.6f}")
    print(f"  Standard error:          {se_rich:.4f}")
    
    if p_rich < 0.05:
        trend_direction = "declining" if slope_rich < 0 else "increasing"
        print(f"  *** Statistically significant {trend_direction} trend (p < 0.05) ***")
    else:
        print(f"  No statistically significant trend (p >= 0.05)")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE_DOUBLE)
    
    # Plot Total Cover trend
    sns.scatterplot(
        x='Year', y='TotalCover', data=yearly_trends,
        ax=axes[0], s=60, color='steelblue'
    )
    axes[0].plot(
        yearly_trends['Year'],
        intercept_cover + slope_cover * yearly_trends['Year'],
        'r-', linewidth=2, label=f'Trend (slope={slope_cover:.3f})'
    )
    axes[0].set_title('Long-Term Trend in Total Coral Cover', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Year', fontsize=10)
    axes[0].set_ylabel('Mean Total Coral Cover (%)', fontsize=10)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Plot Species Richness trend
    sns.scatterplot(
        x='Year', y='SpeciesRichness', data=yearly_trends,
        ax=axes[1], s=60, color='forestgreen'
    )
    axes[1].plot(
        yearly_trends['Year'],
        intercept_rich + slope_rich * yearly_trends['Year'],
        'r-', linewidth=2, label=f'Trend (slope={slope_rich:.3f})'
    )
    axes[1].set_title('Long-Term Trend in Species Richness', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Year', fontsize=10)
    axes[1].set_ylabel('Mean Species Richness', fontsize=10)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_figure:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig_path = os.path.join(FIGURES_DIR, 'longterm_trends.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {fig_path}")
    
    plt.show()
    
    return {
        'yearly_trends': yearly_trends,
        'cover_trend': {
            'slope': slope_cover,
            'intercept': intercept_cover,
            'r_squared': r_cover**2,
            'p_value': p_cover,
            'std_error': se_cover
        },
        'richness_trend': {
            'slope': slope_rich,
            'intercept': intercept_rich,
            'r_squared': r_rich**2,
            'p_value': p_rich,
            'std_error': se_rich
        }
    }


def analyze_net_changes(
    coral_data: pd.DataFrame,
    species_cols: List[str]
) -> Dict:
    """
    Calculate net changes in coral cover and species richness.
    
    Compares the first and last years in the dataset to determine
    overall changes during the monitoring period.
    
    Parameters:
    -----------
    coral_data : pd.DataFrame
        Coral cover data
    species_cols : list
        List of species column names
    
    Returns:
    --------
    dict
        Dictionary containing net change statistics including:
        - first_year, last_year: Year range
        - cover_change: Absolute change in percent cover
        - cover_percent_change: Relative change in percent cover
        - richness_change: Absolute change in species richness
        - first_year_stats, last_year_stats: Mean values for each year
    """
    print("\n" + "=" * 60)
    print("NET CHANGE ANALYSIS")
    print("=" * 60)
    
    if 'TotalCover' not in coral_data.columns:
        coral_data = calculate_coral_metrics(coral_data, species_cols)
    
    first_year = int(coral_data['Year'].min())
    last_year = int(coral_data['Year'].max())
    
    # Calculate statistics for first year
    first_year_data = coral_data[coral_data['Year'] == first_year]
    first_year_stats = {
        'TotalCover': first_year_data['TotalCover'].mean(),
        'TotalCover_std': first_year_data['TotalCover'].std(),
        'SpeciesRichness': first_year_data['SpeciesRichness'].mean(),
        'SpeciesRichness_std': first_year_data['SpeciesRichness'].std(),
        'n_samples': len(first_year_data)
    }
    
    # Calculate statistics for last year
    last_year_data = coral_data[coral_data['Year'] == last_year]
    last_year_stats = {
        'TotalCover': last_year_data['TotalCover'].mean(),
        'TotalCover_std': last_year_data['TotalCover'].std(),
        'SpeciesRichness': last_year_data['SpeciesRichness'].mean(),
        'SpeciesRichness_std': last_year_data['SpeciesRichness'].std(),
        'n_samples': len(last_year_data)
    }
    
    # Calculate changes
    cover_change = last_year_stats['TotalCover'] - first_year_stats['TotalCover']
    if first_year_stats['TotalCover'] > 0:
        cover_percent_change = (cover_change / first_year_stats['TotalCover']) * 100
    else:
        cover_percent_change = 0
    
    richness_change = last_year_stats['SpeciesRichness'] - first_year_stats['SpeciesRichness']
    if first_year_stats['SpeciesRichness'] > 0:
        richness_percent_change = (richness_change / first_year_stats['SpeciesRichness']) * 100
    else:
        richness_percent_change = 0
    
    # Print results
    print(f"\nMonitoring Period: {first_year} to {last_year} ({last_year - first_year} years)")
    
    print("\n" + "-" * 40)
    print(f"FIRST YEAR ({first_year}):")
    print("-" * 40)
    print(f"  Mean Total Cover:      {first_year_stats['TotalCover']:.2f}% "
          f"(± {first_year_stats['TotalCover_std']:.2f})")
    print(f"  Mean Species Richness: {first_year_stats['SpeciesRichness']:.2f} "
          f"(± {first_year_stats['SpeciesRichness_std']:.2f})")
    print(f"  Number of samples:     {first_year_stats['n_samples']}")
    
    print("\n" + "-" * 40)
    print(f"LAST YEAR ({last_year}):")
    print("-" * 40)
    print(f"  Mean Total Cover:      {last_year_stats['TotalCover']:.2f}% "
          f"(± {last_year_stats['TotalCover_std']:.2f})")
    print(f"  Mean Species Richness: {last_year_stats['SpeciesRichness']:.2f} "
          f"(± {last_year_stats['SpeciesRichness_std']:.2f})")
    print(f"  Number of samples:     {last_year_stats['n_samples']}")
    
    print("\n" + "-" * 40)
    print("NET CHANGES:")
    print("-" * 40)
    print(f"  Total Cover Change:      {cover_change:+.2f}% ({cover_percent_change:+.1f}% relative)")
    print(f"  Species Richness Change: {richness_change:+.2f} species ({richness_percent_change:+.1f}% relative)")
    
    return {
        'first_year': first_year,
        'last_year': last_year,
        'years_monitored': last_year - first_year,
        'cover_change': cover_change,
        'cover_percent_change': cover_percent_change,
        'richness_change': richness_change,
        'richness_percent_change': richness_percent_change,
        'first_year_stats': first_year_stats,
        'last_year_stats': last_year_stats
    }


def analyze_regional_variations(
    coral_data: pd.DataFrame,
    species_cols: List[str],
    save_figure: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze coral cover variations by subregion and habitat.
    
    Examines spatial patterns in coral cover across different
    geographic subregions and habitat types.
    
    Parameters:
    -----------
    coral_data : pd.DataFrame
        Coral cover data
    species_cols : list
        List of species column names
    save_figure : bool
        If True, save figure to outputs/figures/
    
    Returns:
    --------
    tuple
        (subregion_trends, habitat_trends) DataFrames with yearly means
    """
    print("\n" + "=" * 60)
    print("REGIONAL VARIATION ANALYSIS")
    print("=" * 60)
    
    if 'TotalCover' not in coral_data.columns:
        coral_data = calculate_coral_metrics(coral_data, species_cols)
    
    cover_col = 'TotalCover'
    
    # Calculate trends by subregion
    subregion_trends = coral_data.groupby(
        ['Year', 'Subregion']
    )[cover_col].mean().reset_index()
    
    # Calculate trends by habitat
    habitat_trends = coral_data.groupby(
        ['Year', 'Habitat']
    )[cover_col].mean().reset_index()
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIG_SIZE_TALL)
    
    # Plot subregion trends
    sns.lineplot(
        data=subregion_trends, x='Year', y=cover_col,
        hue='Subregion', ax=ax1, marker='o', linewidth=2
    )
    ax1.set_title('Coral Cover Trends by Subregion', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Year', fontsize=10)
    ax1.set_ylabel('Mean Percent Cover', fontsize=10)
    ax1.legend(title='Subregion', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot habitat trends
    sns.lineplot(
        data=habitat_trends, x='Year', y=cover_col,
        hue='Habitat', ax=ax2, marker='s', linewidth=2
    )
    ax2.set_title('Coral Cover Trends by Habitat', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Year', fontsize=10)
    ax2.set_ylabel('Mean Percent Cover', fontsize=10)
    ax2.legend(title='Habitat', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_figure:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig_path = os.path.join(FIGURES_DIR, 'regional_trends.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {fig_path}")
    
    plt.show()
    
    # Print summary statistics by subregion
    print("\n" + "-" * 40)
    print("SUMMARY BY SUBREGION:")
    print("-" * 40)
    regional_summary = coral_data.groupby('Subregion').agg({
        cover_col: ['mean', 'std', 'min', 'max', 'count']
    }).round(2)
    regional_summary.columns = ['Mean', 'Std', 'Min', 'Max', 'N']
    print(regional_summary)
    
    # Print summary statistics by habitat
    print("\n" + "-" * 40)
    print("SUMMARY BY HABITAT:")
    print("-" * 40)
    habitat_summary = coral_data.groupby('Habitat').agg({
        cover_col: ['mean', 'std', 'min', 'max', 'count']
    }).round(2)
    habitat_summary.columns = ['Mean', 'Std', 'Min', 'Max', 'N']
    print(habitat_summary)
    
    # Calculate which subregion has declined most
    print("\n" + "-" * 40)
    print("SUBREGION CHANGES (First vs Last Year):")
    print("-" * 40)
    
    first_year = coral_data['Year'].min()
    last_year = coral_data['Year'].max()
    
    for subregion in coral_data['Subregion'].unique():
        sub_data = coral_data[coral_data['Subregion'] == subregion]
        first_mean = sub_data[sub_data['Year'] == first_year][cover_col].mean()
        last_mean = sub_data[sub_data['Year'] == last_year][cover_col].mean()
        change = last_mean - first_mean
        print(f"  {subregion}: {first_mean:.2f}% → {last_mean:.2f}% ({change:+.2f}%)")
    
    return subregion_trends, habitat_trends


def predict_future_trends(
    coral_data: pd.DataFrame,
    species_cols: List[str],
    years_ahead: int = 10,
    save_figure: bool = False
) -> Dict:
    """
    Build a linear predictive model for future coral cover trends.
    
    Uses historical data to project future coral cover with
    95% prediction intervals.
    
    Parameters:
    -----------
    coral_data : pd.DataFrame
        Coral cover data
    species_cols : list
        List of species column names
    years_ahead : int
        Number of years to predict ahead
    save_figure : bool
        If True, save figure to outputs/figures/
    
    Returns:
    --------
    dict
        Dictionary containing:
        - model: Fitted LinearRegression model
        - r_squared: Model R-squared value
        - slope: Annual rate of change
        - future_years: Array of predicted years
        - future_predictions: Array of predicted values
        - prediction_intervals: 95% prediction interval widths
    """
    print("\n" + "=" * 60)
    print("FUTURE TREND PREDICTIONS")
    print("=" * 60)
    
    if 'TotalCover' not in coral_data.columns:
        coral_data = calculate_coral_metrics(coral_data, species_cols)
    
    # Calculate yearly means
    yearly_means = coral_data.groupby('Year')['TotalCover'].agg(
        ['mean', 'std', 'count']
    ).reset_index()
    yearly_means.columns = ['Year', 'Mean', 'Std', 'N']
    yearly_means = yearly_means.sort_values('Year')
    
    X = yearly_means['Year'].values.reshape(-1, 1)
    y = yearly_means['Mean'].values
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate predictions
    last_year = int(X[-1][0])
    first_year = int(X[0][0])
    future_years = np.arange(
        last_year + 1, last_year + years_ahead + 1
    ).reshape(-1, 1)
    future_predictions = model.predict(future_years)
    
    # Calculate prediction intervals
    prediction_intervals = _calculate_prediction_intervals(
        X.flatten(), y, future_years.flatten(), model
    )
    
    # Print model statistics
    print("\n" + "-" * 40)
    print("MODEL STATISTICS:")
    print("-" * 40)
    print(f"  Training period:       {first_year} - {last_year}")
    print(f"  Number of years:       {len(yearly_means)}")
    print(f"  R-squared:             {r2_score(y, model.predict(X)):.4f}")
    print(f"  Slope (annual change): {model.coef_[0]:.4f}% per year")
    print(f"  Intercept:             {model.intercept_:.4f}")
    print(f"  Current cover ({last_year}):  {y[-1]:.2f}%")
    
    # Print predictions
    print("\n" + "-" * 40)
    print(f"PREDICTIONS ({last_year + 1} - {last_year + years_ahead}):")
    print("-" * 40)
    print(f"  {'Year':<8} {'Predicted':>12} {'95% CI Lower':>14} {'95% CI Upper':>14}")
    print("  " + "-" * 50)
    
    for i in range(years_ahead):
        year = int(future_years[i][0])
        pred = future_predictions[i]
        pi = prediction_intervals[i]
        lower = max(0, pred - pi)  # Cover can't be negative
        upper = pred + pi
        print(f"  {year:<8} {pred:>12.2f}% {lower:>14.2f}% {upper:>14.2f}%")
    
    # Calculate time to critical threshold if declining
    if model.coef_[0] < 0:
        current_cover = y[-1]
        
        print("\n" + "-" * 40)
        print("DECLINE PROJECTIONS:")
        print("-" * 40)
        
        # Years until 50% loss
        years_to_half = abs(current_cover / (2 * model.coef_[0]))
        print(f"  At current rate of {model.coef_[0]:.4f}% per year:")
        print(f"  Years until 50% loss from current: {years_to_half:.1f} years")
        print(f"  Estimated year of 50% loss:        {last_year + years_to_half:.0f}")
        
        # Years until effective loss (< 5% cover)
        if current_cover > 5:
            years_to_critical = abs((current_cover - 5) / model.coef_[0])
            print(f"  Years until < 5% cover:            {years_to_critical:.1f} years")
            print(f"  Estimated year of critical loss:   {last_year + years_to_critical:.0f}")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    
    # Historical data
    ax.scatter(X, y, color='steelblue', s=60, label='Historical Data', zorder=3)
    ax.plot(X, model.predict(X), color='red', linewidth=2,
            label='Historical Trend', zorder=2)
    
    # Future predictions
    ax.plot(future_years, future_predictions, color='green', linestyle='--',
            linewidth=2, label='Future Prediction', zorder=2)
    
    # Prediction intervals
    ax.fill_between(
        future_years.flatten(),
        np.maximum(0, future_predictions - prediction_intervals),
        future_predictions + prediction_intervals,
        color='green', alpha=0.2, label='95% Prediction Interval', zorder=1
    )
    
    # Add reference line at 0
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    # Formatting
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel('Mean Coral Cover (%)', fontsize=10)
    ax.set_title(f'Historical Coral Cover and Future Projections '
                 f'({last_year + 1}-{last_year + years_ahead})',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Set y-axis minimum to 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if save_figure:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig_path = os.path.join(FIGURES_DIR, 'future_predictions.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {fig_path}")
    
    plt.show()
    
    return {
        'model': model,
        'r_squared': r2_score(y, model.predict(X)),
        'slope': model.coef_[0],
        'intercept': model.intercept_,
        'last_historical_year': last_year,
        'last_historical_value': y[-1],
        'future_years': future_years.flatten(),
        'future_predictions': future_predictions,
        'prediction_intervals': prediction_intervals
    }


def _calculate_prediction_intervals(
    x: np.ndarray,
    y: np.ndarray,
    x_new: np.ndarray,
    model: LinearRegression
) -> np.ndarray:
    """
    Calculate 95% prediction intervals for new predictions.
    
    Parameters:
    -----------
    x : array
        Historical x values (years)
    y : array
        Historical y values (coral cover)
    x_new : array
        New x values for prediction
    model : LinearRegression
        Fitted model
    
    Returns:
    --------
    array
        Prediction interval widths (half-widths of 95% CI)
    """
    n = len(x)
    x_mean = np.mean(x)
    
    # Calculate residual standard error
    y_pred = model.predict(x.reshape(-1, 1))
    sum_squared_errors = np.sum((y - y_pred) ** 2)
    residual_std_error = np.sqrt(sum_squared_errors / (n - 2))
    
    # Calculate sum of squares for x
    ss_x = np.sum((x - x_mean) ** 2)
    
    # Standard error of prediction for each new point
    std_error_pred = residual_std_error * np.sqrt(
        1 + 1/n + (x_new - x_mean)**2 / ss_x
    )
    
    # t-value for 95% confidence interval
    t_value = stats.t.ppf(0.975, n - 2)
    
    return t_value * std_error_pred


def process_temperature_data(
    filepath: str,
    chunk_size: int = CHUNK_SIZE
) -> Dict[str, float]:
    """
    Process large temperature data file in chunks.
    
    Uses sum + count accumulation to compute true weighted mean
    across all chunks, avoiding the "average of averages" problem.
    
    Parameters:
    -----------
    filepath : str
        Path to temperature CSV file
    chunk_size : int
        Number of rows per chunk
    
    Returns:
    --------
    dict
        Dictionary of site IDs to average temperatures
    """
    print("\n" + "=" * 60)
    print("TEMPERATURE DATA PROCESSING")
    print("=" * 60)
    print(f"File: {filepath}")
    print(f"Chunk size: {chunk_size:,} rows")
    
    site_temps = {}
    total_rows = 0
    
    try:
        chunks = pd.read_csv(filepath, chunksize=chunk_size)
        
        for i, chunk in enumerate(chunks):
            total_rows += len(chunk)
            
            for site_id, group in chunk.groupby("SiteID")["TempC"]:
                if site_id not in site_temps:
                    site_temps[site_id] = {'sum': 0.0, 'count': 0}
                site_temps[site_id]['sum'] += group.sum()
                site_temps[site_id]['count'] += group.count()
            
            # Progress update every 5 chunks
            if (i + 1) % 5 == 0:
                print(f"  Processed {total_rows:,} rows...")
        
        # Calculate final averages
        final_avg_temps = {
            site: vals['sum'] / vals['count']
            for site, vals in site_temps.items()
            if vals['count'] > 0
        }
        
        print(f"\nCompleted processing:")
        print(f"  Total rows:     {total_rows:,}")
        print(f"  Unique sites:   {len(final_avg_temps)}")
        
        # Print sample of results
        print(f"\nSample temperatures (first 5 sites):")
        for site, temp in list(final_avg_temps.items())[:5]:
            print(f"  {site}: {temp:.2f}°C")
        
        return final_avg_temps
        
    except FileNotFoundError:
        print(f"  ERROR: Temperature file not found: {filepath}")
        return {}
    except Exception as e:
        print(f"  ERROR: Failed to process temperature data: {e}")
        return {}


def generate_analysis_report(
    trend_results: Dict,
    net_changes: Dict,
    predictions: Dict,
    output_path: Optional[str] = None
) -> str:
    """
    Generate a text summary report of all analysis results.
    
    Parameters:
    -----------
    trend_results : dict
        Results from analyze_long_term_trends()
    net_changes : dict
        Results from analyze_net_changes()
    predictions : dict
        Results from predict_future_trends()
    output_path : str, optional
        Path to save the report (uses default if None)
    
    Returns:
    --------
    str
        Path to saved report file
    """
    if output_path is None:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        output_path = os.path.join(REPORTS_DIR, 'analysis_report.txt')
    
    report_lines = [
        "=" * 70,
        "CORAL REEF MONITORING AND ANALYSIS REPORT",
        "=" * 70,
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "-" * 70,
        "1. LONG-TERM TRENDS",
        "-" * 70,
        "",
        "Total Coral Cover:",
        f"  Slope (annual change): {trend_results['cover_trend']['slope']:.4f}% per year",
        f"  R-squared:             {trend_results['cover_trend']['r_squared']:.4f}",
        f"  P-value:               {trend_results['cover_trend']['p_value']:.6f}",
        "",
        "Species Richness:",
        f"  Slope (annual change): {trend_results['richness_trend']['slope']:.4f} species per year",
        f"  R-squared:             {trend_results['richness_trend']['r_squared']:.4f}",
        f"  P-value:               {trend_results['richness_trend']['p_value']:.6f}",
        "",
        "-" * 70,
        "2. NET CHANGES",
        "-" * 70,
        "",
        f"Monitoring Period: {net_changes['first_year']} to {net_changes['last_year']} "
        f"({net_changes['years_monitored']} years)",
        "",
        f"Total Cover Change:      {net_changes['cover_change']:+.2f}% "
        f"({net_changes['cover_percent_change']:+.1f}% relative)",
        f"Species Richness Change: {net_changes['richness_change']:+.2f} species "
        f"({net_changes['richness_percent_change']:+.1f}% relative)",
        "",
        "-" * 70,
        "3. FUTURE PREDICTIONS",
        "-" * 70,
        "",
        f"Model R-squared:         {predictions['r_squared']:.4f}",
        f"Annual rate of change:   {predictions['slope']:.4f}% per year",
        "",
        "Projected coral cover:",
    ]
    
    for i in range(0, len(predictions['future_years']), 3):
        year = int(predictions['future_years'][i])
        pred = predictions['future_predictions'][i]
        pi = predictions['prediction_intervals'][i]
        report_lines.append(
            f"  {year}: {pred:.2f}% (95% CI: {max(0, pred-pi):.2f}% - {pred+pi:.2f}%)"
        )
    
    report_lines.extend([
        "",
        "=" * 70,
        "END OF REPORT",
        "=" * 70
    ])
    
    report_text = "\n".join(report_lines)
    
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(f"\nAnalysis report saved to: {output_path}")
    return output_path


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function for coral reef data analysis.
    
    Runs the complete analysis pipeline:
    1. Load coral cover data
    2. Audit all data files
    3. Analyze long-term trends
    4. Calculate net changes
    5. Analyze regional variations
    6. Generate future predictions
    7. Process temperature data (if available)
    8. Generate summary report
    """
    print("=" * 70)
    print("CORAL REEF MONITORING AND ANALYSIS")
    print("=" * 70)
    print(f"Author: Cece Foltz")
    print(f"Date: 03/31/2025")
    print("=" * 70)
    
    # Ensure directories exist
    ensure_directories()
    
    print(f"\nCurrent working directory: {os.getcwd()}")
    print(f"Cleaned data directory: {CLEANED_DATA_DIR}")
    print(f"Figures directory: {FIGURES_DIR}")
    print(f"Reports directory: {REPORTS_DIR}")
    
    # =========================================================================
    # STEP 1: Load primary coral cover data
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)
    
    coral_data = find_and_load_csv(
        'CREMP_Pcover_2023_StonyCoralSpecies_Cleaned.csv'
    )
    
    if coral_data is None:
        # Try loading raw data if cleaned not found
        print("\nCleaned data not found. Trying raw data...")
        coral_data = find_and_load_csv(
            'CREMP_Pcover_2023_StonyCoralSpecies.csv',
            search_cleaned_first=False
        )
    
    if coral_data is None:
        print("\nERROR: Could not load coral data. Exiting.")
        print("Please ensure data files are in the correct directory.")
        return
    
    print_dataframe_summary(coral_data, "Coral Cover Data")
    
    # =========================================================================
    # STEP 2: Identify species columns
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: IDENTIFYING SPECIES COLUMNS")
    print("=" * 70)
    
    species_cols = get_species_columns(coral_data)
    print(f"\nIdentified {len(species_cols)} species columns")
    print(f"First 10 species: {species_cols[:10]}")
    if len(species_cols) > 10:
        print(f"... and {len(species_cols) - 10} more")
    
    # =========================================================================
    # STEP 3: Audit all data files
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: AUDITING DATA FILES")
    print("=" * 70)
    
    audit_results = {}
    for filename in DATA_FILES:
        # Try cleaned directory first, then raw
        filepath = os.path.join(CLEANED_DATA_DIR, filename)
        if not os.path.exists(filepath):
            filepath = os.path.join(RAW_DATA_DIR, filename)
        if not os.path.exists(filepath):
            filepath = filename
        
        if os.path.exists(filepath):
            result = analyze_missing_and_outliers(filepath)
            if result:
                audit_results[filename] = result
        else:
            print(f"File not found: {filename}")
    
    print(f"\nAudited {len(audit_results)} of {len(DATA_FILES)} files")
    
    # =========================================================================
    # STEP 4: Long-term trend analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: LONG-TERM TREND ANALYSIS")
    print("=" * 70)
    
    trend_results = analyze_long_term_trends(
        coral_data.copy(),
        species_cols,
        save_figure=True
    )
    
    # =========================================================================
    # STEP 5: Net change analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: NET CHANGE ANALYSIS")
    print("=" * 70)
    
    net_changes = analyze_net_changes(
        coral_data.copy(),
        species_cols
    )
    
    # =========================================================================
    # STEP 6: Regional variation analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: REGIONAL VARIATION ANALYSIS")
    print("=" * 70)
    
    subregion_trends, habitat_trends = analyze_regional_variations(
        coral_data.copy(),
        species_cols,
        save_figure=True
    )
    
    # =========================================================================
    # STEP 7: Future predictions
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: FUTURE PREDICTIONS")
    print("=" * 70)
    
    predictions = predict_future_trends(
        coral_data.copy(),
        species_cols,
        years_ahead=10,
        save_figure=True
    )
    
    # =========================================================================
    # STEP 8: Process temperature data (if available)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 8: TEMPERATURE DATA PROCESSING")
    print("=" * 70)
    
    # Try multiple possible locations for temperature file
    temp_file = None
    temp_paths = [
        os.path.join(RAW_DATA_DIR, 'CREMP_Temperatures_2023.csv'),
        os.path.join(CLEANED_DATA_DIR, 'CREMP_Temperatures_2023.csv'),
        'CREMP_Temperatures_2023.csv'
    ]
    
    for path in temp_paths:
        if os.path.exists(path):
            temp_file = path
            break
    
    if temp_file:
        site_temperatures = process_temperature_data(temp_file)
    else:
        print("\nTemperature file not found.")
        print("This file is too large for GitHub and must be downloaded separately.")
        print("See README.md for instructions.")
        site_temperatures = {}
    
    # =========================================================================
    # STEP 9: Generate summary report
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 9: GENERATING SUMMARY REPORT")
    print("=" * 70)
    
    report_path = generate_analysis_report(
        trend_results,
        net_changes,
        predictions
    )
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    print("\nKey Findings:")
    print("-" * 40)
    
    # Cover trend summary
    cover_slope = trend_results['cover_trend']['slope']
    cover_p = trend_results['cover_trend']['p_value']
    if cover_p < 0.05:
        direction = "declining" if cover_slope < 0 else "increasing"
        print(f"• Coral cover is significantly {direction} "
              f"at {abs(cover_slope):.3f}% per year (p={cover_p:.4f})")
    else:
        print(f"• No significant trend in coral cover (p={cover_p:.4f})")
    
    # Net change summary
    print(f"• Net change from {net_changes['first_year']} to {net_changes['last_year']}: "
          f"{net_changes['cover_change']:+.2f}%")
    
    # Prediction summary
    last_pred_year = int(predictions['future_years'][-1])
    last_pred_value = predictions['future_predictions'][-1]
    print(f"• Projected cover in {last_pred_year}: {last_pred_value:.2f}%")
    
    print("\nOutput Files:")
    print("-" * 40)
    print(f"• Figures:  {FIGURES_DIR}")
    print(f"• Report:   {report_path}")
    
    print("\n" + "=" * 70)
    print("Thank you for using the Coral Reef Monitoring Analysis Tool")
    print("=" * 70)


if __name__ == "__main__":
    main()
