import os

import geopandas
import matplotlib
import numpy
import pandas
import pandas as pd
import pip
import seaborn
from geopy.geocoders import arcgis

# Print current working directory and list files to help debug
print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir())

# Try to find the CSV file in parent directories
try:
    # First try the current directory
    if os.path.exists('CREMP_Pcover_2023_StonyCoralSpecies.csv'):
        coral_data = pd.read_csv('CREMP_Pcover_2023_StonyCoralSpecies.csv')
    # Try parent directory
    elif os.path.exists('../CREMP_Pcover_2023_StonyCoralSpecies.csv'):
        coral_data = pd.read_csv('../CREMP_Pcover_2023_StonyCoralSpecies.csv')
    # Try parent of parent directory
    elif os.path.exists('../../CREMP_Pcover_2023_StonyCoralSpecies.csv'):
        coral_data = pd.read_csv('../../CREMP_Pcover_2023_StonyCoralSpecies.csv')
    # Last resort - specify the full path that you believe is correct
    else:
        coral_data = pd.read_csv('C:/Users/User/data.csv/CoralReefML/CREMP_Pcover_2023_StonyCoralSpecies.csv')

    # If any of the above succeeded, print the first 20 rows
    print(coral_data.head(20))
    print("Loaded 'CREMP_Pcover_2023_StonyCoralSpecies.csv' and displayed the first 20 rows.")

except FileNotFoundError as e:
    print("Error finding the CSV file:", e)
    print("\nPlease ensure the CSV file exists and try one of these solutions:")
    print("1. Move the CSV file to:", os.getcwd())
    print("2. Update the path in your code to point to where the CSV file is located")
    import pandas as pd
    import numpy as np
    import os

    # Define the list of uploaded files
    files = [
        'CREMP_OCTO_RawData_2023.csv',
        'CREMP_OCTO_Summaries_2023_Density.csv',
        'CREMP_OCTO_Summaries_2023_MeanHeight.csv',
        'CREMP_Pcover_2023_StonyCoralSpecies.csv',
        'CREMP_Pcover_2023_TaxaGroups.csv',
        'CREMP_SCOR_RawData_2023.csv',
        'CREMP_SCOR_Summaries_2023_ConditionCounts.csv',
        'CREMP_SCOR_Summaries_2023_Counts.csv',
        'CREMP_SCOR_Summaries_2023_Density.csv',
        'CREMP_SCOR_Summaries_2023_LTA.csv',
        'CREMP_Stations_2023.csv',
        'CREMP_Temperatures_2023.csv'
    ]


    def analyze_missing_outliers(file):
        print("-" * 50)
        print("Analyzing file: " + file)

        try:
            df = pd.read_csv(file)
        except Exception as e:
            print("Error reading file: " + str(e))
            return

            # Print basic shape and head for context
        print("Shape:", df.shape)
        print("Head:\n", df.head())

        # Missing values count per column
        missing_summary = df.isnull().sum()
        print("\nMissing values per column:\n", missing_summary[missing_summary > 0])

        # For numeric columns, identify outliers using the IQR method
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print("\nOutlier summary (IQR method for numeric variables):")
            outlier_summary = {}
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                outlier_summary[col] = len(outliers)
            for col, count in outlier_summary.items():
                print(col + ":", count, "outliers")
        else:
            print("\nNo numeric columns to check for outliers.")

            # Also check datatypes for potential inconsistencies
        print("\nData types:\n", df.dtypes)

        print("\nFinished analysis for file:", file)
        print("=" * 50, "\n")

        # Iterate through each file and analyze


    for f in files:
        if os.path.exists(f):
            analyze_missing_outliers(f)
        else:
            print("File not found: " + f)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load coral reef data (assuming it's already loaded as coral_data from previous cell)
# If not, load it:
# coral_data = pd.read_csv('CREMP_Pcover_2023_StonyCoralSpecies.csv')

# Identify species columns (assume metadata columns known):
metadata_cols = ['OID_', 'Year', 'Date', 'Subregion', 'Habitat', 'SiteID', 'Site_name', 'StationID',
                 'Surveyed_all_years', 'points']

species_cols = [col for col in coral_data.columns if col not in metadata_cols]

# Calculate total percent cover (assuming values represent percent cover) and species richness
coral_data['TotalCover'] = coral_data[species_cols].sum(axis=1)
coral_data['SpeciesRichness'] = (coral_data[species_cols] > 0).sum(axis=1)

# Group by Year for long-term trends
yearly_trends = coral_data.groupby('Year').agg({'TotalCover': 'mean', 'SpeciesRichness': 'mean'}).reset_index()

print('Yearly trends (mean):')
print(yearly_trends.head())

# Perform linear regression for total cover trend vs Year
slope_cover, intercept_cover, r_value_cover, p_value_cover, std_err_cover = stats.linregress(yearly_trends['Year'],
                                                                                             yearly_trends[
                                                                                                 'TotalCover'])

# Perform linear regression for species richness trend vs Year
slope_rich, intercept_rich, r_value_rich, p_value_rich, std_err_rich = stats.linregress(yearly_trends['Year'],
                                                                                        yearly_trends[
                                                                                            'SpeciesRichness'])

print("\
Long-term trend for Total Coral Cover:")
print("Slope: " + str(slope_cover))
print("P-value: " + str(p_value_cover))

print("\
Long-term trend for Species Richness:")
print("Slope: " + str(slope_rich))
print("P-value: " + str(p_value_rich))

# Plot the trends
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Plot Total Coral Cover trend
sns.scatterplot(x='Year', y='TotalCover', data=yearly_trends, ax=ax[0])
ax[0].plot(yearly_trends['Year'], intercept_cover + slope_cover * yearly_trends['Year'], 'r', label='Fit')
ax[0].set_title('Long-Term Trend in Total Coral Cover')
ax[0].set_ylabel('Mean Total Coral Cover (%)')
ax[0].legend()

# Plot Species Richness trend
sns.scatterplot(x='Year', y='SpeciesRichness', data=yearly_trends, ax=ax[1])
ax[1].plot(yearly_trends['Year'], intercept_rich + slope_rich * yearly_trends['Year'], 'r', label='Fit')
ax[1].set_title('Long-Term Trend in Species Richness')
ax[1].set_ylabel('Mean Species Richness')
ax[1].legend()

plt.tight_layout()
plt.show()

print("\
Trend analysis complete.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the stony coral data
coral_data = pd.read_csv('CREMP_Pcover_2023_StonyCoralSpecies.csv')

# Calculate species richness and total cover for each survey
species_cols = [col for col in coral_data.columns if col not in ['OID_', 'Year', 'Date', 'Subregion', 'Habitat', 'SiteID', 'Site_name', 'StationID', 'Surveyed_all_years', 'points']]
coral_data['SpeciesRichness'] = (coral_data[species_cols] > 0).sum(axis=1)
coral_data['TotalCover'] = coral_data[species_cols].sum(axis=1)

# Calculate annual means across all stations
annual_trends = coral_data.groupby('Year').agg({
    'TotalCover': ['mean', 'std', 'count'],
    'SpeciesRichness': ['mean', 'std', 'count']
}).round(2)

# Calculate subregion trends
subregion_trends = coral_data.groupby(['Year', 'Subregion']).agg({
    'TotalCover': 'mean',
    'SpeciesRichness': 'mean'
}).reset_index()

# Perform linear regression for overall trend
years = coral_data['Year'].unique()
annual_cover = coral_data.groupby('Year')['TotalCover'].mean()
annual_richness = coral_data.groupby('Year')['SpeciesRichness'].mean()

slope_cover, intercept_cover, r_value_cover, p_value_cover, std_err_cover = stats.linregress(years, annual_cover)
slope_richness, intercept_richness, r_value_richness, p_value_richness, std_err_richness = stats.linregress(years, annual_richness)

# Create visualization of trends
plt.figure(figsize=(15, 10))

# Plot 1: Total Cover Trends
plt.subplot(2, 1, 1)
sns.regplot(data=coral_data, x='Year', y='TotalCover', scatter=False, color='blue', line_kws={'linestyle': '--'})
sns.boxplot(data=coral_data, x='Year', y='TotalCover', color='lightblue')
plt.title('Long-term Trends in Total Stony Coral Cover')
plt.ylabel('Total Cover (%)')

# Plot 2: Species Richness Trends
plt.subplot(2, 1, 2)
sns.regplot(data=coral_data, x='Year', y='SpeciesRichness', scatter=False, color='green', line_kws={'linestyle': '--'})
sns.boxplot(data=coral_data, x='Year', y='SpeciesRichness', color='lightgreen')
plt.title('Long-term Trends in Species Richness')
plt.ylabel('Number of Species')

plt.tight_layout()
plt.show()

# Print statistical results
print("\
Long-term Trend Analysis Results:")
print("\
Total Cover:")
print(f"Annual rate of change: {slope_cover:.3f}% per year")
print(f"R-squared: {r_value_cover**2:.3f}")
print(f"P-value: {p_value_cover:.4f}")

print("\
Species Richness:")
print(f"Annual rate of change: {slope_richness:.3f} species per year")
print(f"R-squared: {r_value_richness**2:.3f}")
print(f"P-value: {p_value_richness:.4f}")

# Calculate net changes between first and last year
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

print(f"\
Net Changes ({first_year} to {last_year}):")
print(f"Total Cover: {(last_year_stats['TotalCover'] - first_year_stats['TotalCover']):.2f}%")
print(f"Species Richness: {(last_year_stats['SpeciesRichness'] - first_year_stats['SpeciesRichness']):.2f} species")
import pandas as pd

# Load the coral reef data file. Here we use the 'CREMP_Pcover_2023_StonyCoralSpecies.csv' file.
coral_data = pd.read_csv('CREMP_Pcover_2023_StonyCoralSpecies.csv')

# Display the first 20 rows
print(coral_data.head(20))

print("Loaded 'CREMP_Pcover_2023_StonyCoralSpecies.csv' and displayed the first 20 rows.")
import pandas as pd
import numpy as np
import os

# Define the list of uploaded files
files = [
    'CREMP_OCTO_RawData_2023.csv',
    'CREMP_OCTO_Summaries_2023_Density.csv',
    'CREMP_OCTO_Summaries_2023_MeanHeight.csv',
    'CREMP_Pcover_2023_StonyCoralSpecies.csv',
    'CREMP_Pcover_2023_TaxaGroups.csv',
    'CREMP_SCOR_RawData_2023.csv',
    'CREMP_SCOR_Summaries_2023_ConditionCounts.csv',
    'CREMP_SCOR_Summaries_2023_Counts.csv',
    'CREMP_SCOR_Summaries_2023_Density.csv',
    'CREMP_SCOR_Summaries_2023_LTA.csv',
    'CREMP_Stations_2023.csv',
    'CREMP_Temperatures_2023.csv'
]


def analyze_missing_outliers(file):
    print("-" * 50)
    print("Analyzing file: " + file)

    try:
        df = pd.read_csv(file)
    except Exception as e:
        print("Error reading file: " + str(e))
        return

        # Print basic shape and head for context
    print("Shape:", df.shape)
    print("Head:\n", df.head())

    # Missing values count per column
    missing_summary = df.isnull().sum()
    print("\nMissing values per column:\n", missing_summary[missing_summary > 0])

    # For numeric columns, identify outliers using the IQR method
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print("\nOutlier summary (IQR method for numeric variables):")
        outlier_summary = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outlier_summary[col] = len(outliers)
        for col, count in outlier_summary.items():
            print(col + ":", count, "outliers")
    else:
        print("\nNo numeric columns to check for outliers.")

        # Also check datatypes for potential inconsistencies
    print("\nData types:\n", df.dtypes)

    print("\nFinished analysis for file:", file)
    print("=" * 50, "\n")


# Iterate through each file and analyze
for f in files:
    if os.path.exists(f):
        analyze_missing_outliers(f)
    else:
        print("File not found: " + f)
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy import stats

        # Load coral reef data (assuming it's already loaded as coral_data from previous cell)
        # If not, load it:
        # coral_data = pd.read_csv('CREMP_Pcover_2023_StonyCoralSpecies.csv')

        # Identify species columns (assume metadata columns known):
        metadata_cols = ['OID_', 'Year', 'Date', 'Subregion', 'Habitat', 'SiteID', 'Site_name', 'StationID',
                         'Surveyed_all_years', 'points']

        species_cols = [col for col in coral_data.columns if col not in metadata_cols]

        # Calculate total percent cover (assuming values represent percent cover) and species richness
        coral_data['TotalCover'] = coral_data[species_cols].sum(axis=1)
        coral_data['SpeciesRichness'] = (coral_data[species_cols] > 0).sum(axis=1)

        # Group by Year for long-term trends
        yearly_trends = coral_data.groupby('Year').agg({'TotalCover': 'mean', 'SpeciesRichness': 'mean'}).reset_index()

        print('Yearly trends (mean):')
        print(yearly_trends.head())

        # Perform linear regression for total cover trend vs Year
        slope_cover, intercept_cover, r_value_cover, p_value_cover, std_err_cover = stats.linregress(
            yearly_trends['Year'], yearly_trends['TotalCover'])

        # Perform linear regression for species richness trend vs Year
        slope_rich, intercept_rich, r_value_rich, p_value_rich, std_err_rich = stats.linregress(yearly_trends['Year'],
                                                                                                yearly_trends[
                                                                                                    'SpeciesRichness'])

        print("\
        Long-term trend for Total Coral Cover:")
        print("Slope: " + str(slope_cover))
        print("P-value: " + str(p_value_cover))

        print("\
        Long-term trend for Species Richness:")
        print("Slope: " + str(slope_rich))
        print("P-value: " + str(p_value_rich))

        # Plot the trends
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))

        # Plot Total Coral Cover trend
        sns.scatterplot(x='Year', y='TotalCover', data=yearly_trends, ax=ax[0])
        ax[0].plot(yearly_trends['Year'], intercept_cover + slope_cover * yearly_trends['Year'], 'r', label='Fit')
        ax[0].set_title('Long-Term Trend in Total Coral Cover')
        ax[0].set_ylabel('Mean Total Coral Cover (%)')
        ax[0].legend()

        # Plot Species Richness trend
        sns.scatterplot(x='Year', y='SpeciesRichness', data=yearly_trends, ax=ax[1])
        ax[1].plot(yearly_trends['Year'], intercept_rich + slope_rich * yearly_trends['Year'], 'r', label='Fit')
        ax[1].set_title('Long-Term Trend in Species Richness')
        ax[1].set_ylabel('Mean Species Richness')
        ax[1].legend()

        plt.tight_layout()
        plt.show()

        print("\
        Trend analysis complete.")
chunksize = 50000  # Adjust as needed
chunk_list = []  # Store chunks here

# Read the data in chunks
for chunk in pd.read_csv('CREMP_OCTO_RawData_2023.csv', chunksize=chunksize):
    chunk_list.append(chunk)

# Concatenate chunks after processing
coral_data = pd.concat(chunk_list, axis=0)
chunk = pd.read_csv("CREMP_Temperatures_2023.csv", nrows=100)
print(chunk.head())
import pandas as pd

chunk_size = 100_000  # Adjust this as needed
chunks = pd.read_csv("CREMP_Temperatures_2023.csv", chunksize=chunk_size)

# Example: preview the first chunk
first_chunk = next(chunks)
print(first_chunk.head())
import pandas as pd

# Set the chunk size to a manageable number (e.g., 100,000 rows)
chunk_size = 100_000
chunks = pd.read_csv("CREMP_Temperatures_2023.csv", chunksize=chunk_size)

# Example: Process each chunk (e.g., calculating average temperature per SiteID)
site_avg_temps = {}

for chunk in chunks:
    # Group by 'SiteID' and calculate mean temperature
    grouped = chunk.groupby("SiteID")["TempC"].mean()

    # Store the result in a dictionary (site_id -> avg_temp)
    for site_id, avg_temp in grouped.items():
        if site_id not in site_avg_temps:
            site_avg_temps[site_id] = []
        site_avg_temps[site_id].append(avg_temp)

# Calculate overall average temperature per site
final_avg_temp = {site: sum(temps) / len(temps) for site, temps in site_avg_temps.items()}

print(final_avg_temp)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the stony coral percent cover data
pcover_df = pd.read_csv('CREMP_Pcover_2023_StonyCoralSpecies.csv')

# Get coral species columns (excluding metadata columns)
coral_species = pcover_df.columns[10:]  # All columns after metadata

# Calculate total coral cover per site/year
yearly_cover = pcover_df.groupby('Year')[coral_species].mean().sum(axis=1)

# Calculate species richness (number of species with cover > 0)
yearly_richness = pcover_df.groupby('Year')[coral_species].apply(lambda x: (x > 0).sum(axis=1).mean())

# Calculate net changes
start_year = yearly_cover.index.min()
end_year = yearly_cover.index.max()

# Percent cover changes
start_cover = yearly_cover[start_year]
end_cover = yearly_cover[end_year]
net_change_cover = end_cover - start_cover
percent_change_cover = (net_change_cover/start_cover) * 100

# Species richness changes
start_richness = yearly_richness[start_year]
end_richness = yearly_richness[end_year]
net_change_richness = end_richness - start_richness
percent_change_richness = (net_change_richness/start_richness) * 100

print('Reef Community Parameter Changes from', start_year, 'to', end_year)
print('\
Stony Coral Cover:')
print('Initial cover: {:.2f}%'.format(start_cover))
print('Final cover: {:.2f}%'.format(end_cover))
print('Net change: {:.2f}%'.format(net_change_cover))
print('Percent change: {:.1f}%'.format(percent_change_cover))

print('\
Species Richness:')
print('Initial mean species per site: {:.2f}'.format(start_richness))
print('Final mean species per site: {:.2f}'.format(end_richness))
print('Net change: {:.2f} species'.format(net_change_richness))
print('Percent change: {:.1f}%'.format(percent_change_richness))

# Visualize trends
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Coral Cover Trend
sns.regplot(x=yearly_cover.index, y=yearly_cover.values,
            scatter=True, ci=95, ax=ax1)
ax1.set_title('Stony Coral Cover Trend')
ax1.set_xlabel('Year')
ax1.set_ylabel('Total Percent Cover')

# Species Richness Trend
sns.regplot(x=yearly_richness.index, y=yearly_richness.values,
            scatter=True, ci=95, ax=ax2)
ax2.set_title('Species Richness Trend')
ax2.set_xlabel('Year')
ax2.set_ylabel('Mean Number of Species per Site')

plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the datasets
pcover_df = pd.read_csv('CREMP_Pcover_2023_StonyCoralSpecies.csv')
stations_df = pd.read_csv('CREMP_Stations_2023.csv')

# Get coral species columns
coral_species = pcover_df.columns[10:]

# Calculate total coral cover per site
pcover_df['total_cover'] = pcover_df[coral_species].sum(axis=1)
pcover_df['species_richness'] = (pcover_df[coral_species] > 0).sum(axis=1)

# 1. Analyze variations by subregion and habitat
subregion_trends = pcover_df.groupby(['Year', 'Subregion'])['total_cover'].mean().reset_index()
habitat_trends = pcover_df.groupby(['Year', 'Habitat'])['total_cover'].mean().reset_index()

# Create subplot for regional and habitat trends
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

# Plot subregional trends
sns.lineplot(data=subregion_trends, x='Year', y='total_cover', hue='Subregion', ax=ax1)
ax1.set_title('Coral Cover Trends by Subregion')
ax1.set_ylabel('Mean Percent Cover')

# Plot habitat trends
sns.lineplot(data=habitat_trends, x='Year', y='total_cover', hue='Habitat', ax=ax2)
ax2.set_title('Coral Cover Trends by Habitat')
ax2.set_ylabel('Mean Percent Cover')

plt.tight_layout()
plt.show()

# Calculate statistical summaries
print("Regional Changes in Coral Cover:")
regional_summary = pcover_df.groupby('Subregion').agg({
    'total_cover': ['mean', 'std', 'min', 'max']
}).round(2)
print(regional_summary)

# Calculate net change by subregion
early_years = pcover_df[pcover_df['Year'] <= 2000]
recent_years = pcover_df[pcover_df['Year'] >= 2020]

early_mean = early_years.groupby('Subregion')['total_cover'].mean()
recent_mean = recent_years.groupby('Subregion')['total_cover'].mean()
percent_change = ((recent_mean - early_mean) / early_mean * 100).round(1)

print("\
Percent Change by Subregion (2000 vs 2020+):")
print(percent_change)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load and prepare data
pcover_df = pd.read_csv('CREMP_Pcover_2023_StonyCoralSpecies.csv')
coral_species = pcover_df.columns[10:]
pcover_df['total_cover'] = pcover_df[coral_species].sum(axis=1)

# Calculate annual means for more stable predictions
yearly_means = pcover_df.groupby('Year')['total_cover'].agg(['mean', 'std']).reset_index()

# Prepare data for modeling
X = yearly_means['Year'].values.reshape(-1, 1)
y = yearly_means['mean'].values

# Create and fit linear model
model = LinearRegression()
model.fit(X, y)

# Generate future predictions (10 years into the future)
future_years = np.arange(X[-1][0] + 1, X[-1][0] + 11).reshape(-1, 1)
future_predictions = model.predict(future_years)


# Calculate confidence intervals
def prediction_interval(x, y, x_new):
    n = len(x)
    x_mean = np.mean(x)

    # Sum of squared errors
    sum_squared_errors = np.sum((y - model.predict(x.reshape(-1, 1))) ** 2)
    std_error = np.sqrt(sum_squared_errors / (n - 2))

    # Standard error of prediction
    x_new_mean = np.mean(x_new)
    std_error_pred = std_error * np.sqrt(1 + 1 / n +
                                         (x_new - x_mean) ** 2 /
                                         np.sum((x - x_mean) ** 2))

    # 95% prediction interval
    t_value = stats.t.ppf(0.975, n - 2)
    pi = t_value * std_error_pred

    return pi


# Calculate prediction intervals
pi = prediction_interval(X.flatten(), y, future_years.flatten())

# Plot historical data and future predictions with confidence intervals
plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='blue', label='Historical Data')
plt.plot(X, model.predict(X), color='red', label='Historical Trend')
plt.plot(future_years, future_predictions, color='green', linestyle='--', label='Future Prediction')
plt.fill_between(future_years.flatten(),
                 future_predictions - pi,
                 future_predictions + pi,
                 color='green', alpha=0.2, label='95% Prediction Interval')

plt.xlabel('Year')
plt.ylabel('Mean Coral Cover (%)')
plt.title('Historical Coral Cover and Future Projections (2024-2033)')
plt.legend()
plt.grid(True)
plt.show()

# Print model statistics and predictions
print("Model Statistics:")
print(f"R-squared: {r2_score(y, model.predict(X)):.3f}")
print(f"Slope (annual change): {model.coef_[0]:.4f}")
print(f"Current coral cover (2023): {y[-1]:.2f}%")
print("\
Projected coral cover for key years:")
for i, year in enumerate(future_years[::3].flatten()):
    pred_index = i * 3
    print(
        f"Year {int(year)}: {future_predictions[pred_index]:.2f}% (95% PI: {future_predictions[pred_index] - pi[pred_index]:.2f}% to {future_predictions[pred_index] + pi[pred_index]:.2f}%)")

# Calculate time to critical thresholds
if model.coef_[0] < 0:  # Only if declining trend
    current_cover = y[-1]
    years_to_half = abs(current_cover / (2 * model.coef_[0]))
    print(f"\
At current rate of decline:")
    print(f"Years until 50% loss from current cover: {years_to_half:.1f} years")
    print(f"Estimated year of 50% loss: {2023 + years_to_half:.0f}")

 

