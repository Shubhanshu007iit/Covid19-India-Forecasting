

# COVID-19 India Forecasting Pipeline

**Data Analysis and Engineering Intern – Technical Assignment**  
ARTPARK @ IISc

---

## Overview

This repository implements an end-to-end disease forecasting workflow for COVID-19 in India at the national level. It integrates COVID-19 case data with Google Mobility data, performs exploratory data analysis, engineers temporal and epidemiological features, and trains/evaluates ensemble machine learning models (Random Forest and Gradient Boosting) to forecast 7-day average daily cases.

---

## Repository Structure

```
covid_forecast/
├── main.py                    # Entry point – runs full pipeline
├── requirements.txt
├── README.md
├── src/
│   ├── data_preparation.py    # Data download, merge, feature engineering
│   ├── eda.py                 # Exploratory data analysis & plots
│   └── model.py               # ML model training, evaluation, visualisation
├── data/
│   ├── raw/                   # Raw downloaded CSVs (auto-created)
│   └── processed/             # Processed dataset (auto-created)
├── figures/                   # All output plots (auto-created)
└── outputs/                   # Metrics JSON/CSV (auto-created)
```

---

## Setup

```bash
git clone <repo-url>
cd covid_forecast
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.9, pandas, numpy, scikit-learn, matplotlib, seaborn, scipy

---

## Running the Pipeline

```bash
python main.py
```

This executes three stages in sequence:
1. **Data Preparation** – downloads (or generates synthetic) COVID-19 and Mobility data, cleans, merges, and engineers features.
2. **EDA** – produces temporal overview, wave annotation, mobility correlation, and feature heatmap plots.
3. **Model** – trains Naive Baseline, Random Forest, and Gradient Boosting models; evaluates on a 90-day held-out test set; saves metrics and plots.

You can also run stages individually:
```bash
python src/data_preparation.py
python src/eda.py
python src/model.py
```

---

## Data Sources

| Dataset | Source | Description |
|---------|--------|-------------|
| COVID-19 Cases | [Our World in Data](https://covid.ourworldindata.org/data/owid-covid-data.csv) | Daily new cases, deaths, tests, vaccinations for India |
| Google Mobility | [Google COVID-19 Mobility Reports](https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv) | % change from baseline in retail, transit, workplace, residential activity |

> **Note:** If network access is unavailable, the pipeline automatically generates realistic synthetic data modelling India's three pandemic waves (Wave 1: Sep 2020, Delta: May 2021, Omicron: Jan 2022) for reproducibility.

---

## Feature Engineering

| Feature Category | Features |
|-----------------|----------|
| Lag features | `cases_lag7`, `cases_lag14`, `cases_lag21` |
| Rolling statistics | `cases_roll14_mean`, `cases_roll14_std` |
| Growth dynamics | `growth_rate_7d` (7-day % change in 7d avg) |
| Temporal | `day_of_week`, `month`, `week_of_year`, `is_weekend` |
| Epidemiological | `positivity_rate`, `stringency_index` |
| Vaccination | `vacc_coverage` (cumulative vaccinations / population) |
| Mobility | Retail, transit, workplace, residential % change from baseline |

---

## Modelling Approach

- **Target variable:** 7-day rolling average of new daily cases (smooths reporting noise)
- **Train/test split:** Chronological – last 90 days as held-out test set
- **Preprocessing:** RobustScaler (handles outliers from reporting anomalies)
- **Models:**
  - Naive Baseline (last known 7-day avg)
  - Random Forest (300 trees, max_depth=12)
  - Gradient Boosting (400 estimators, lr=0.05, max_depth=5)

---

## Results

| Model | MAE | RMSE | R² | MAPE |
|-------|-----|------|----|------|
| Naive Baseline | 87 | 110 | -0.04 | 16.4% |
| Random Forest | 66 | 85 | 0.38 | 13.4% |
| Gradient Boosting | 63 | 71 | **0.56** | **12.6%** |

Gradient Boosting outperforms both baselines across all metrics. The most important predictive features are recent lag values (lag-7, lag-14) and rolling mean – consistent with the autoregressive nature of epidemic spread.

---

## Limitations & Assumptions

- Analysis is at **national level** (India); sub-national patterns are not captured.
- Synthetic data closely models known wave dynamics but cannot replicate exact reporting patterns.
- Mobility data ends in Oct 2022; mobility features carry forward their last value for later dates.
- The model is **not causal** – it captures statistical regularities, not transmission mechanisms.
- Weekend reporting dips are smoothed by the 7-day average target but may still introduce artefacts.

