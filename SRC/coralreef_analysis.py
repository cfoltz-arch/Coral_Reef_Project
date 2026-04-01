"""
Coral Reef Monitoring and Analysis Script
=========================================
This script analyzes CREMP (Coral Reef Evaluation and Monitoring Project) data
to assess coral reef health trends, species richness, and future projections.

Author: Cece Foltz
Date: 03/31/2026

Cleaning decisions applied in this script:
- Duplicates removed from raw files
- Missing species values left as NaN (reflects unobserved/unsurveyed taxa,
  not confirmed absence — imputing 0 would distort means)
- IQR outlier removal applied ONLY to unbounded physical measurements
  (Diameter_cm, Height_cm). NOT applied to percent cover, counts, density,
  LTA, or condition data — these are zero-inflated ecological distributions
  where IQR would flag genuine presence records as outliers
- Negative LTA values removed (physically impossible)
- TotalCover calculated with min_count=1 so all-NaN rows return NaN, not 0
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# =============================================================================
# CONFIGURATION
# =============================================================================

# Set your local data directory here if files are not in the script's folder.
# Example: DATA_DIR = 'C:/Users/Cece/Documents/CoralReefML/data/cleaned/'
DATA_DIR = ''

# Cleaned file names (output of data cleaning pipeline)
DATA_FILES = [
    'CREMP_OCTO_Cleaned_2023.csv',
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
    'CREMP_Temperatures_2023.csv'  # Too large for GitHub — must be placed locally
]

# Metadata columns that are not species data
METADATA_COLS = [
    'OID_', 'Year', 'Date', 'Subregion', 'Habitat',
    'SiteID', 'Site_name', 'StationID', 'Surveyed_all_years', 'points'
]

# Chunk size for processing large files (e.g. temperature data)
CHUNK_SIZE = 100_000


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def find_and_load_csv(filename, data_dir=DATA_DIR):
    """
    Search for and load a CSV file from multiple possible locations.

    Parameters:
    -----------
    filename : str
        Name of the CSV file to load
    data_dir : str
        Primary directory to search first (set DATA_DIR at top of script)

    Returns:
    --------
    pd.DataFrame or None
        Loaded dataframe or None if file not found
    """
    paths_to_try = [
        os.path.join(data_dir, filename) if data_dir else filename,
        filename,
        f'../{filename}',
        f'../../{filename}',
        f'data/cleaned/{filename}',
        f'data/{filename}',
    ]

    for path in paths_to_try:
        if os.path.exists(path):
            print(f"Found file at: {path}")
            return pd.read_csv(path)

    print(f"Error: Could not find {filename}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir()}")
    return None


def analyze_missing_and_outliers(filepath):
    """
    Audit a cleaned CSV file for remaining missing values and outlier flags.

    NOTE: Outlier counts here are for informational purposes only.
    IQR is not an appropriate removal criterion for zero-inflated
    species matrices — see script header for full reasoning.

    Parameters:
    -----------
    filepath : str
        Path to the CSV file

    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    print("-" * 50)
    print(f"Analyzing file: {filepath}")

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

    # Missing values — expected in species columns, left as NaN intentionally
    missing_summary = df.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]
    if len(missing_cols) > 0:
        print(f"\nMissing values per column (left as NaN — not imputed):\n{missing_cols}")
        results['missing_values'] = missing_cols.to_dict()
    else:
        print("\nNo missing values.")

    # Outlier audit — informational only, not used for removal
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

    print(f"\nData types:\n{df.dtypes}")
    print("=" * 50 + "\n")

    return results


def get_species_columns(df, metadata_cols=METADATA_COLS):
    """
    Extract species column names from dataframe by excluding known metadata columns.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    metadata_cols : list
        List of metadata column names to exclude

    Returns:
    --------
    list
        List of species column names
    """
    return [col for col in df.columns if col not in metadata_cols]


def calculate_coral_metrics(df, species_cols):
    """
    Calculate total cover and species richness for coral data.

    TotalCover uses min_count=1 so that rows where ALL species are NaN
    return NaN rather than 0. Rows with a mix of real values and NaN
    are summed over observed values only — consistent with leaving
    missing species as NaN rather than imputing 0.

    Parameters:
    -----------
    df : pd.DataFrame
        Coral cover dataframe
    species_cols : list
        List of species column names

    Returns:
    --------
    pd.DataFrame
        Dataframe with added TotalCover and SpeciesRichness columns
    """
    df = df.copy()
    df['TotalCover'] = df[species_cols].sum(axis=1, min_count=1)
    df['SpeciesRichness'] = (df[species_cols] > 0).sum(axis=1)
    return df


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_long_term_trends(coral_data, species_cols):
    """
    Perform long-term trend analysis on coral cover and species richness.

    Parameters:
    -----------
    coral_data : pd.DataFrame
        Coral cover data with Year column
    species_cols : list
        List of species column names

    Returns:
    --------
    dict
        Dictionary containing trend analysis results
    """
    if 'TotalCover' not in coral_data.columns:
        coral_data = calculate_coral_metrics(coral_data, species_cols)

    # NaN TotalCover rows are excluded from groupby mean automatically
    yearly_trends = coral_data.groupby('Year').agg({
        'TotalCover': 'mean',
        'SpeciesRichness': 'mean'
    }).reset_index()

    # Sort ensures linregress receives years in order
    yearly_trends = yearly_trends.sort_values('Year')

    print("Yearly trends (mean):")
    print(yearly_trends.head(10))

    slope_cover, intercept_cover, r_cover, p_cover, se_cover = stats.linregress(
        yearly_trends['Year'], yearly_trends['TotalCover']
    )
    slope_rich, intercept_rich, r_rich, p_rich, se_rich = stats.linregress(
        yearly_trends['Year'], yearly_trends['SpeciesRichness']
    )

    print("\nLong-term trend for Total Coral Cover:")
    print(f"  Slope: {slope_cover:.4f}")
    print(f"  R-squared: {r_cover**2:.4f}")
    print(f"  P-value: {p_cover:.4f}")

    print("\nLong-term trend for Species Richness:")
    print(f"  Slope: {slope_rich:.4f}")
    print(f"  R-squared: {r_rich**2:.4f}")
    print(f"  P-value: {p_rich:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.scatterplot(x='Year', y='TotalCover', data=yearly_trends, ax=axes[0])
    axes[0].plot(
        yearly_trends['Year'],
        intercept_cover + slope_cover * yearly_trends['Year'],
        'r', label=f'Fit (slope={slope_cover:.3f})'
    )
    axes[0].set_title('Long-Term Trend in Total Coral Cover')
    axes[0].set_ylabel('Mean Total Coral Cover (%)')
    axes[0].legend()

    sns.scatterplot(x='Year', y='SpeciesRichness', data=yearly_trends, ax=axes[1])
    axes[1].plot(
        yearly_trends['Year'],
        intercept_rich + slope_rich * yearly_trends['Year'],
        'r', label=f'Fit (slope={slope_rich:.3f})'
    )
    axes[1].set_title('Long-Term Trend in Species Richness')
    axes[1].set_ylabel('Mean Species Richness')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return {
        'yearly_trends': yearly_trends,
        'cover_trend': {
            'slope': slope_cover,
            'intercept': intercept_cover,
            'r_squared': r_cover**2,
            'p_value': p_cover
        },
        'richness_trend': {
            'slope': slope_rich,
            'intercept': intercept_rich,
            'r_squared': r_rich**2,
            'p_value': p_rich
        }
    }


def analyze_net_changes(coral_data, species_cols):
    """
    Calculate net changes in coral cover and species richness between first and last year.

    Parameters:
    -----------
    coral_data : pd.DataFrame
        Coral cover data
    species_cols : list
        List of species column names

    Returns:
    --------
    dict
        Dictionary containing net change statistics
    """
    if 'TotalCover' not in coral_data.columns:
        coral_data = calculate_coral_metrics(coral_data, species_cols)

    first_year = coral_data['Year'].min()
    last_year = coral_data['Year'].max()

    first_year_stats = coral_data[coral_data['Year'] == first_year].agg({
        'TotalCover': 'mean',
        'SpeciesRichness': 'mean'
    })

    last_year_stats = coral_data[coral_data['Year'] == last_year].agg({
        'TotalCover': 'mean',
        'SpeciesRichness': 'mean'
    })

    cover_change = last_year_stats['TotalCover'] - first_year_stats['TotalCover']
    richness_change = last_year_stats['SpeciesRichness'] - first_year_stats['SpeciesRichness']

    print(f"\nNet Changes ({first_year} to {last_year}):")
    print(f"  Total Cover: {cover_change:.2f}%")
    print(f"  Species Richness: {richness_change:.2f} species")

    return {
        'first_year': first_year,
        'last_year': last_year,
        'cover_change': cover_change,
        'richness_change': richness_change,
        'first_year_stats': first_year_stats.to_dict(),
        'last_year_stats': last_year_stats.to_dict()
    }


def analyze_regional_variations(coral_data, species_cols):
    """
    Analyze coral cover variations by subregion and habitat.

    Parameters:
    -----------
    coral_data : pd.DataFrame
        Coral cover data
    species_cols : list
        List of species column names

    Returns:
    --------
    tuple
        (subregion_trends, habitat_trends) DataFrames
    """
    if 'TotalCover' not in coral_data.columns:
        coral_data = calculate_coral_metrics(coral_data, species_cols)

    cover_col = 'TotalCover'

    subregion_trends = coral_data.groupby(['Year', 'Subregion'])[cover_col].mean().reset_index()
    habitat_trends = coral_data.groupby(['Year', 'Habitat'])[cover_col].mean().reset_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    sns.lineplot(data=subregion_trends, x='Year', y=cover_col, hue='Subregion', ax=ax1)
    ax1.set_title('Coral Cover Trends by Subregion')
    ax1.set_ylabel('Mean Percent Cover')

    sns.lineplot(data=habitat_trends, x='Year', y=cover_col, hue='Habitat', ax=ax2)
    ax2.set_title('Coral Cover Trends by Habitat')
    ax2.set_ylabel('Mean Percent Cover')

    plt.tight_layout()
    plt.show()

    print("\nRegional Changes in Coral Cover:")
    regional_summary = coral_data.groupby('Subregion').agg({
        cover_col: ['mean', 'std', 'min', 'max']
    }).round(2)
    print(regional_summary)

    return subregion_trends, habitat_trends


def predict_future_trends(coral_data, species_cols, years_ahead=10):
    """
    Build a linear predictive model for future coral cover trends.

    Parameters:
    -----------
    coral_data : pd.DataFrame
        Coral cover data
    species_cols : list
        List of species column names
    years_ahead : int
        Number of years to predict ahead

    Returns:
    --------
    dict
        Dictionary containing model and predictions
    """
    if 'TotalCover' not in coral_data.columns:
        coral_data = calculate_coral_metrics(coral_data, species_cols)

    yearly_means = coral_data.groupby('Year')['TotalCover'].agg(['mean', 'std']).reset_index()
    yearly_means = yearly_means.sort_values('Year')

    X = yearly_means['Year'].values.reshape(-1, 1)
    y = yearly_means['mean'].values

    model = LinearRegression()
    model.fit(X, y)

    last_year = int(X[-1][0])
    future_years = np.arange(last_year + 1, last_year + years_ahead + 1).reshape(-1, 1)
    future_predictions = model.predict(future_years)

    prediction_intervals = _calculate_prediction_intervals(
        X.flatten(), y, future_years.flatten(), model
    )

    plt.figure(figsize=(12, 6))
    plt.scatter(X, y, color='blue', label='Historical Data', alpha=0.7)
    plt.plot(X, model.predict(X), color='red', label='Historical Trend', linewidth=2)
    plt.plot(future_years, future_predictions, color='green', linestyle='--',
             label='Future Prediction', linewidth=2)
    plt.fill_between(
        future_years.flatten(),
        future_predictions - prediction_intervals,
        future_predictions + prediction_intervals,
        color='green', alpha=0.2, label='95% Prediction Interval'
    )
    plt.xlabel('Year')
    plt.ylabel('Mean Coral Cover (%)')
    plt.title(f'Historical Coral Cover and Future Projections ({last_year + 1}-{last_year + years_ahead})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print("\nModel Statistics:")
    print(f"  R-squared: {r2_score(y, model.predict(X)):.3f}")
    print(f"  Slope (annual change): {model.coef_[0]:.4f}% per year")
    print(f"  Current coral cover ({last_year}): {y[-1]:.2f}%")

    print(f"\nProjected coral cover for key years:")
    for i in range(0, years_ahead, 3):
        year = int(future_years[i][0])
        pred = future_predictions[i]
        pi = prediction_intervals[i]
        print(f"  Year {year}: {pred:.2f}% (95% PI: {pred - pi:.2f}% to {pred + pi:.2f}%)")

    if model.coef_[0] < 0:
        current_cover = y[-1]
        years_to_half = abs(current_cover / (2 * model.coef_[0]))
        print(f"\nAt current rate of decline:")
        print(f"  Years until 50% loss from current cover: {years_to_half:.1f} years")
        print(f"  Estimated year of 50% loss: {last_year + years_to_half:.0f}")

    return {
        'model': model,
        'r_squared': r2_score(y, model.predict(X)),
        'slope': model.coef_[0],
        'future_years': future_years.flatten(),
        'future_predictions': future_predictions,
        'prediction_intervals': prediction_intervals
    }


def _calculate_prediction_intervals(x, y, x_new, model):
    """
    Calculate 95% prediction intervals for new predictions.

    Parameters:
    -----------
    x : array
        Historical x values
    y : array
        Historical y values
    x_new : array
        New x values for prediction
    model : LinearRegression
        Fitted model

    Returns:
    --------
    array
        Prediction interval widths
    """
    n = len(x)
    x_mean = np.mean(x)

    y_pred = model.predict(x.reshape(-1, 1))
    sum_squared_errors = np.sum((y - y_pred) ** 2)
    std_error = np.sqrt(sum_squared_errors / (n - 2))

    std_error_pred = std_error * np.sqrt(
        1 + 1/n + (x_new - x_mean)**2 / np.sum((x - x_mean)**2)
    )

    t_value = stats.t.ppf(0.975, n - 2)
    return t_value * std_error_pred


def process_temperature_data(filepath, chunk_size=CHUNK_SIZE):
    """
    Process large temperature data file in chunks.

    Uses sum + count accumulation per site across chunks to compute
    the true weighted mean — averaging chunk means would be incorrect
    when chunks contain unequal numbers of rows per site.

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
    print(f"Processing temperature data from {filepath}...")

    site_temps = {}
    chunks = pd.read_csv(filepath, chunksize=chunk_size)

    for chunk in chunks:
        for site_id, group in chunk.groupby("SiteID")["TempC"]:
            if site_id not in site_temps:
                site_temps[site_id] = {'sum': 0, 'count': 0}
            site_temps[site_id]['sum'] += group.sum()
            site_temps[site_id]['count'] += group.count()

    # Divide total sum by total count — not an average of averages
    final_avg_temps = {
        site: vals['sum'] / vals['count']
        for site, vals in site_temps.items()
    }

    print(f"Processed temperatures for {len(final_avg_temps)} sites")
    return final_avg_temps


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function for coral reef data analysis.
    """
    print("=" * 60)
    print("CORAL REEF MONITORING AND ANALYSIS")
    print("=" * 60)

    print(f"\nCurrent working directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir()}")

    # Step 1: Load primary coral cover data (cleaned version)
    print("\n" + "-" * 40)
    print("Loading coral cover data...")
    coral_data = find_and_load_csv('CREMP_Pcover_2023_StonyCoralSpecies_Cleaned.csv')

    if coral_data is None:
        print("ERROR: Could not load coral data. Exiting.")
        return

    print(f"\nData loaded successfully!")
    print(f"Shape: {coral_data.shape}")
    print(f"\nFirst 20 rows:")
    print(coral_data.head(20))

    # Step 2: Identify species columns
    species_cols = get_species_columns(coral_data)
    print(f"\nIdentified {len(species_cols)} species columns")

    # Step 3: Audit all cleaned data files
    print("\n" + "-" * 40)
    print("Auditing all cleaned data files...")
    for filename in DATA_FILES:
        filepath = os.path.join(DATA_DIR, filename) if DATA_DIR else filename
        if os.path.exists(filepath):
            analyze_missing_and_outliers(filepath)
        else:
            print(f"File not found: {filepath}")

    # Step 4: Long-term trend analysis
    print("\n" + "-" * 40)
    print("Performing long-term trend analysis...")
    trend_results = analyze_long_term_trends(coral_data.copy(), species_cols)

    # Step 5: Net change analysis
    print("\n" + "-" * 40)
    print("Calculating net changes...")
    net_changes = analyze_net_changes(coral_data.copy(), species_cols)

    # Step 6: Regional variation analysis
    print("\n" + "-" * 40)
    print("Analyzing regional variations...")
    subregion_trends, habitat_trends = analyze_regional_variations(coral_data.copy(), species_cols)

    # Step 7: Future predictions
    print("\n" + "-" * 40)
    print("Building predictive model...")
    predictions = predict_future_trends(coral_data.copy(), species_cols, years_ahead=10)

    # Step 8: Process temperature data if available locally
    temp_file = os.path.join(DATA_DIR, 'CREMP_Temperatures_2023.csv') if DATA_DIR else 'CREMP_Temperatures_2023.csv'
    if os.path.exists(temp_file):
        print("\n" + "-" * 40)
        print("Processing temperature data...")
        site_temperatures = process_temperature_data(temp_file)
        print(f"Sample temperature data: {dict(list(site_temperatures.items())[:5])}")
    else:
        print("\nTemperature file not found locally — skipping. See README for details.")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
