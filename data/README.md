# Data Directory

This directory contains sample datasets for time series forecasting examples.

## Sample Datasets

The example scripts generate their own synthetic data for demonstration purposes. However, you can place your own datasets here.

## Data Format

For best compatibility with the utility functions, use the following CSV format:

```csv
date,value
2020-01-01,100.5
2020-01-02,102.3
2020-01-03,98.7
...
```

### Requirements:
- First column: Date/timestamp (any format parseable by pandas)
- Subsequent columns: Numerical values

## Using Your Own Data

To use your own data with the examples:

```python
from utils.data_preprocessing import load_time_series_data

# Load your data
df = load_time_series_data('data/your_data.csv', 
                          date_column='date', 
                          value_column='value')
```

## Data Sources

Here are some popular sources for time series data:

### Financial Data
- Yahoo Finance
- Quandl
- Alpha Vantage

### Economic Data
- World Bank
- IMF Data
- FRED (Federal Reserve Economic Data)

### Weather Data
- NOAA
- Weather Underground
- OpenWeatherMap

### Energy Data
- EIA (U.S. Energy Information Administration)
- ENTSO-E (European Network of Transmission System Operators)

### Competition Datasets
- Kaggle Time Series Competitions
- UCI Machine Learning Repository
- M4 Competition

## Data Preprocessing Tips

1. **Check for missing values**: Use `handle_missing_values()`
2. **Check stationarity**: Use `check_stationarity()`
3. **Normalize if needed**: Use `normalize_data()` for neural networks
4. **Split properly**: Use `train_test_split_temporal()` to maintain temporal order

## Example Datasets Structure

```
data/
├── README.md (this file)
├── sample_stock_prices.csv
├── sample_sales_data.csv
├── sample_temperature.csv
└── your_data.csv
```

Place your `.csv` files here, but note that data files are excluded from git by default (see `.gitignore`).
