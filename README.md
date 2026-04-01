# Coral Reef Monitoring and Analysis

A data cleaning and analysis pipeline for the Florida Keys Coral Reef Evaluation and Monitoring Project (CREMP), examining long-term trends in coral reef health, species richness, octocoral density, and future coral cover projections.

**Author:** Cece Foltz  
**Last Updated:** March 31, 2026

---

## Project Overview

This project processes and analyzes multi-decade CREMP monitoring data collected across three Florida Keys subregions (Upper Keys, Middle Keys, Lower Keys). The pipeline covers data cleaning, exploratory analysis, regional trend comparisons, and a predictive model for future stony coral cover.

---

## Repository Structure

```
Coral_Reef_Project/
│
├── coralreef_analysis.py                          # Main analysis script
│
├── data/                                          # Raw input data 
│   ├── CREMP_OCTO_RawData_2023.csv
│   ├── CREMP_OCTO_Summaries_2023_Density.csv
│   ├── CREMP_OCTO_Summaries_2023_MeanHeight.csv
│   ├── CREMP_Pcover_2023_StonyCoralSpecies.csv
│   ├── CREMP_Pcover_2023_TaxaGroups.csv
│   ├── CREMP_SCOR_RawData_2023.csv
│   ├── CREMP_SCOR_Summaries_2023_ConditionCounts.csv
│   ├── CREMP_SCOR_Summaries_2023_Counts.csv
│   ├── CREMP_SCOR_Summaries_2023_Density.csv
│   ├── CREMP_SCOR_Summaries_2023_LTA.csv
│   ├── CREMP_Stations_2023.csv
│   └── CREMP_Temperatures_2023.csv
│
├── cleaned/                                       # Cleaned output files
│   ├── CREMP_OCTO_Cleaned_2023.csv
│   ├── CREMP_OCTO_Summaries_2023_Density_Cleaned.csv
│   └── CREMP_Pcover_2023_StonyCoralSpecies_Cleaned.csv
│
├── schema.ini                                     # Schema definitions for CSV files
└── README.md
```

---

## Data Cleaning Summary

All cleaning was performed in Python using pandas. The consistent approach across all files was:

- **Duplicates** — exact duplicate rows removed
- **Missing values** — left as `NaN` rather than imputed, to preserve ecological meaning (a missing species value reflects absence of a survey observation, not a lost measurement)
- **Outliers** — removed using the IQR method (1.5×IQR) where appropriate; not applied to zero-inflated percent cover data where true ecological presence records would be incorrectly flagged

### File-by-file results

| File | Original Rows | Final Rows | Duplicates Removed | Outliers Removed | Notes |
|---|---|---|---|---|---|
| `CREMP_OCTO_RawData_2023.csv` | 109,246 | 51,499 | 57,028 | 719 | 2 missing `Height_cm` left as NaN |
| `CREMP_OCTO_Summaries_2023_Density.csv` | 1,023 | 542 | 0 | 481 | Species density NaNs left as NaN |
| `CREMP_Pcover_2023_StonyCoralSpecies.csv` | 3,918 | 3,918 | 0 | 0 | IQR not applied; 6 missing dates left as NaN; 112 all-zero rows flagged |

### Why IQR was not applied to percent cover data

The stony coral species columns are zero-inflated ecological proportions — the majority of species are absent at most survey stations, which is ecologically expected rather than anomalous. Applying IQR to these distributions would flag genuine species presence records as outliers and remove real biological signal from the dataset. Values are also bounded between 0 and 1 by definition, so no true out-of-range outliers exist.

---

## Analysis Pipeline

The main script (`coralreef_analysis.py`) performs the following steps in order:

1. **File discovery and loading** — searches multiple relative paths for data files
2. **Missing value and outlier audit** — runs across all 12 CREMP files and reports findings
3. **Long-term trend analysis** — linear regression on annual mean coral cover and species richness
4. **Net change calculation** — compares first and last survey year statistics
5. **Regional variation analysis** — trends broken down by subregion and habitat type
6. **Future projections** — linear model with 95% prediction intervals for 10-year forecast
7. **Temperature processing** — chunk-based processing of temperature records with correct weighted averaging per site

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
```

Install dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

Python 3.8 or higher recommended.

---

## Usage

1. Place all CREMP data files in the same directory as the script, or in a `data/` subdirectory
2. Run the script:

```bash
python coralreef_analysis.py
```

The script will automatically search for data files in the current directory and common relative paths. If a file is not found, it will print the current working directory and skip that file gracefully.

---

## Data Source

Data from the Florida Fish and Wildlife Conservation Commission (FWC) Coral Reef Evaluation and Monitoring Project (CREMP). Survey data spans 1996–2023 across Upper, Middle, and Lower Florida Keys subregions. GIS data from NOAA. 

---

## Notes

- The `all_species_zero` column added to the cleaned stony coral file flags survey stations where no coral cover was recorded. These rows are retained in the dataset but can be filtered out for analyses that require confirmed coral presence.
- Temperature averaging across chunked reads uses sum and count accumulation per site to ensure the correct weighted mean rather than an average of chunk means.
- Missing species density values in the octocoral summaries file should be treated as unobserved rather than zero in any downstream calculations.
