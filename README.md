# 🪸 Coral Reef Monitoring and Analysis

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive data analysis toolkit for monitoring coral reef health using CREMP (Coral Reef Evaluation and Monitoring Project) data from the Florida Keys. This project analyzes long-term trends in coral cover, species richness, and provides predictive modeling for future reef conditions.

![Coral Reef Analysis](docs/images/coral_banner.png)

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Data Cleaning Philosophy](#-data-cleaning-philosophy)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Data](#-data)
- [Analysis Results](#-analysis-results)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## 🔬 Project Overview

This project provides tools for:

- **Data Cleaning**: Ecologically-appropriate cleaning that respects zero-inflated species distributions
- **Long-term Trend Analysis**: Statistical analysis of coral cover and species richness over time
- **Regional Comparisons**: Analysis by subregion (Upper Keys, Middle Keys, Lower Keys, Dry Tortugas) and habitat type
- **Predictive Modeling**: Future projections with confidence intervals
- **Temperature Integration**: Correlation analysis with water temperature data

## 🧹 Data Cleaning Philosophy

This project applies **ecologically-informed** data cleaning decisions:

| Data Type | Cleaning Applied | Rationale |
|-----------|------------------|-----------|
| **Missing species values** | Left as NaN | Represents unobserved taxa, not confirmed absence. Imputing 0 would distort means. |
| **Physical measurements** (Diameter, Height) | IQR outlier removal | Unbounded measurements where extremes likely indicate errors |
| **Percent cover** | NO outlier removal | Zero-inflated, bounded (0-100); low values are ecologically valid |
| **Counts/Density** | NO outlier removal | Zero-inflated ecological data; zeros are genuine absences |
| **LTA (Living Tissue Area)** | Remove negatives only | Negative values are physically impossible |
| **Duplicates** | Removed | Standard cleaning |

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip or conda package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/coral-reef-monitoring.git
cd coral-reef-monitoring

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
