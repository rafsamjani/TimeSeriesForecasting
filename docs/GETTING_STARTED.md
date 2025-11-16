# Getting Started with Time Series Forecasting

This guide will help you get started with the Time Series Forecasting repository, especially if you're working on a thesis (skripsi) or academic research.

## Table of Contents

1. [Understanding Your Data](#understanding-your-data)
2. [Choosing the Right Method](#choosing-the-right-method)
3. [Step-by-Step Workflow](#step-by-step-workflow)
4. [Common Pitfalls](#common-pitfalls)
5. [Tips for Academic Work](#tips-for-academic-work)

## Understanding Your Data

Before applying any forecasting method, understand your data:

### 1. Data Characteristics

Ask yourself:
- **How much data do I have?** (Small: <100 points, Medium: 100-1000, Large: >1000)
- **What's the frequency?** (Daily, weekly, monthly, yearly)
- **Is there seasonality?** (Regular patterns that repeat)
- **Is there a trend?** (Long-term increase or decrease)
- **Are there outliers?** (Unusual values)
- **Is there missing data?**

### 2. Visual Inspection

```python
from utils.visualization import plot_time_series
from utils.data_preprocessing import load_time_series_data

# Load and visualize
data = load_time_series_data('your_data.csv')
plot_time_series(data)
```

### 3. Statistical Tests

```python
from utils.data_preprocessing import check_stationarity

# Check if data is stationary
result = check_stationarity(data)
print(f"Is stationary: {result['is_stationary']}")
print(f"P-value: {result['p_value']}")
```

## Choosing the Right Method

Use this decision tree:

### Small Dataset (<100 observations)
- ✅ Traditional Methods (Moving Average, Exponential Smoothing)
- ✅ ARIMA
- ❌ LSTM (needs more data)
- ⚠️ Prophet (possible but not ideal)

### Medium Dataset (100-1000 observations)
- ✅ ARIMA
- ✅ Prophet
- ✅ Traditional Methods
- ⚠️ LSTM (can work with proper validation)

### Large Dataset (>1000 observations)
- ✅ All methods
- ✅ LSTM particularly good for complex patterns
- ✅ Prophet for multiple seasonality

### Strong Seasonality
- ✅ Prophet (best for multiple seasonality)
- ✅ Holt-Winters (Triple Exponential Smoothing)
- ✅ Seasonal ARIMA (SARIMA)

### Non-linear Patterns
- ✅ LSTM
- ✅ Prophet
- ⚠️ ARIMA (may struggle)

### Need Interpretability
- ✅ ARIMA (coefficients are interpretable)
- ✅ Traditional Methods (simple to explain)
- ⚠️ Prophet (some interpretability)
- ❌ LSTM (black box)

## Step-by-Step Workflow

### Step 1: Load and Explore Data

```python
from utils.data_preprocessing import load_time_series_data
from utils.visualization import plot_time_series, plot_decomposition

# Load data
data = load_time_series_data('data/your_data.csv')

# Visualize
plot_time_series(data)
plot_decomposition(data['value'])
```

### Step 2: Preprocess Data

```python
from utils.data_preprocessing import handle_missing_values, check_stationarity

# Handle missing values
data = handle_missing_values(data, method='interpolate')

# Check stationarity
stationarity = check_stationarity(data, column='value')
print(stationarity)
```

### Step 3: Split Data

```python
from utils.data_preprocessing import train_test_split_temporal

# Split data (80% train, 20% test)
train, test = train_test_split_temporal(data, test_size=0.2)
```

### Step 4: Apply Multiple Methods

```python
# Try different methods
from examples.arima_example import fit_arima_model
from examples.traditional_methods import triple_exponential_smoothing

# ARIMA
arima_forecast = fit_arima_model(train['value'])

# Holt-Winters
hw_forecast = triple_exponential_smoothing(train['value'], test['value'])
```

### Step 5: Evaluate and Compare

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

# Compare models
arima_metrics = evaluate(test['value'], arima_forecast)
hw_metrics = evaluate(test['value'], hw_forecast)
```

### Step 6: Visualize Results

```python
from utils.visualization import plot_multiple_forecasts

forecasts = [arima_forecast, hw_forecast]
labels = ['ARIMA', 'Holt-Winters']

plot_multiple_forecasts(test['value'], forecasts, labels)
```

## Common Pitfalls

### 1. Not Checking for Stationarity
- **Problem**: ARIMA requires stationary data
- **Solution**: Use `check_stationarity()` and apply differencing if needed

### 2. Data Leakage
- **Problem**: Using future data to predict the past
- **Solution**: Always use `train_test_split_temporal()` to maintain temporal order

### 3. Overfitting
- **Problem**: Model works well on training but fails on test data
- **Solution**: Use proper validation, monitor test performance

### 4. Wrong Train/Test Split
- **Problem**: Random splitting breaks temporal structure
- **Solution**: Always split chronologically (earliest dates for training)

### 5. Ignoring Seasonality
- **Problem**: Poor forecasts for seasonal data
- **Solution**: Use methods that handle seasonality (Prophet, SARIMA, Holt-Winters)

### 6. Not Scaling Data for Neural Networks
- **Problem**: LSTM training fails or converges slowly
- **Solution**: Use `normalize_data()` before training

## Tips for Academic Work

### For Your Thesis/Skripsi

1. **Literature Review**
   - Research similar studies in your domain
   - Justify your choice of methods
   - Cite relevant papers

2. **Methodology Section**
   - Clearly describe each method used
   - Explain why you chose these methods
   - Document all parameters and hyperparameters

3. **Results Section**
   - Present metrics in tables
   - Include visualizations
   - Compare multiple methods
   - Discuss which method performed best and why

4. **Reproducibility**
   - Document your Python version
   - List all package versions (use `requirements.txt`)
   - Provide clear steps to reproduce results
   - Share your code (like this repository!)

### Structuring Your Code

```python
# Good practice for thesis code
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting analysis...")
    
    # 1. Load data
    logger.info("Loading data...")
    data = load_data()
    
    # 2. Preprocess
    logger.info("Preprocessing...")
    data = preprocess(data)
    
    # 3. Model training
    logger.info("Training models...")
    models = train_models(data)
    
    # 4. Evaluation
    logger.info("Evaluating...")
    results = evaluate_models(models, test_data)
    
    # 5. Save results
    logger.info("Saving results...")
    save_results(results)
    
    logger.info("Analysis complete!")

if __name__ == '__main__':
    main()
```

### Creating Publication-Ready Figures

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

# Create figure
fig, ax = plt.subplots()
ax.plot(data, linewidth=2)
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('Your Title Here')
plt.tight_layout()
plt.savefig('figure1.png', dpi=300, bbox_inches='tight')
```

### Statistical Reporting

Always report:
- Sample size (n)
- Time period covered
- Evaluation metrics with confidence intervals if possible
- Comparison with baseline methods
- Statistical significance tests

### Example Results Table

| Method | RMSE | MAE | MAPE (%) | Training Time |
|--------|------|-----|----------|---------------|
| ARIMA | 5.23 | 4.11 | 3.45 | 2.3s |
| LSTM | 4.89 | 3.87 | 3.12 | 45.6s |
| Prophet | 5.45 | 4.23 | 3.67 | 8.9s |

## Next Steps

1. **Run all examples** to understand each method
2. **Apply to your data** using the workflow above
3. **Document everything** for your thesis
4. **Compare results** systematically
5. **Iterate and improve** based on results

## Need Help?

- Check the main [README.md](../README.md)
- Review example code in `examples/`
- Look at utility functions in `utils/`
- Open an issue on GitHub

Good luck with your research! 🎓📊
