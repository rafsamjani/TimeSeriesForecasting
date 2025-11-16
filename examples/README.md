# Examples Directory

This directory contains complete, runnable examples of different time series forecasting methods.

## Available Examples

### 1. ARIMA Example (`arima_example.py`)

**What it does:**
- Generates sample time series data with trend and seasonality
- Fits an ARIMA(1,1,1) model
- Generates forecasts
- Evaluates performance
- Creates visualizations (ACF/PACF, forecast, residuals)

**Run it:**
```bash
python examples/arima_example.py
```

**Output files:**
- `arima_acf_pacf.png` - Autocorrelation analysis
- `arima_forecast.png` - Forecast comparison
- `arima_residuals.png` - Residual diagnostics

**Use case:** 
- Stationary or trend-stationary data
- Economic indicators
- Financial time series
- Academic research requiring statistical rigor

---

### 2. LSTM Example (`lstm_example.py`)

**What it does:**
- Generates daily time series data
- Prepares sequences for LSTM
- Builds and trains LSTM model
- Generates multi-step forecasts
- Evaluates performance

**Run it:**
```bash
python examples/lstm_example.py
```

**Output files:**
- `lstm_training_history.png` - Training/validation loss
- `lstm_forecast.png` - Forecast comparison

**Use case:**
- Large datasets (1000+ observations)
- Complex non-linear patterns
- Multivariate forecasting
- When traditional methods fail

**Note:** Requires TensorFlow/Keras installation.

---

### 3. Prophet Example (`prophet_example.py`)

**What it does:**
- Generates daily time series with multiple seasonality
- Fits Prophet model
- Generates forecasts with confidence intervals
- Decomposes into trend and seasonal components
- Evaluates performance

**Run it:**
```bash
python examples/prophet_example.py
```

**Output files:**
- `prophet_forecast_full.png` - Complete forecast visualization
- `prophet_components.png` - Trend and seasonal components
- `prophet_forecast.png` - Test set comparison
- `prophet_residuals.png` - Residual diagnostics

**Use case:**
- Daily business metrics
- Data with holidays
- Multiple seasonal patterns
- Robust handling of missing data

**Note:** Requires Prophet installation.

---

### 4. Traditional Methods (`traditional_methods.py`)

**What it does:**
- Demonstrates 6 classical forecasting methods:
  - Moving Average
  - Simple Exponential Smoothing
  - Double Exponential Smoothing (Holt's method)
  - Triple Exponential Smoothing (Holt-Winters)
  - Naive Forecast
  - Seasonal Naive Forecast
- Compares all methods side-by-side
- Evaluates performance

**Run it:**
```bash
python examples/traditional_methods.py
```

**Output files:**
- `traditional_methods_comparison.png` - All methods compared
- `traditional_methods_individual.png` - Individual method plots

**Use case:**
- Baseline comparisons
- Simple patterns
- Educational purposes
- Quick prototyping

---

## Running All Examples

To run all examples at once:

```bash
cd examples
python arima_example.py
python traditional_methods.py
# python lstm_example.py  # Takes longer
# python prophet_example.py  # Requires additional package
```

## Customizing Examples

Each example is self-contained and easy to modify:

### Use Your Own Data

Replace the `load_sample_data()` function:

```python
def load_sample_data():
    """Load your own data"""
    df = pd.read_csv('your_data.csv', parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df
```

### Change Parameters

Find the model fitting section and adjust parameters:

```python
# ARIMA
model = ARIMA(train_data, order=(2, 1, 2))  # Changed from (1,1,1)

# LSTM
model = build_lstm_model(seq_length, units=100)  # Changed from 50

# Prophet
model = Prophet(changepoint_prior_scale=0.1)  # Changed from 0.05
```

### Modify Visualization

All plotting code is clearly marked:

```python
# Change figure size
plt.figure(figsize=(20, 10))  # Changed from (14, 7)

# Change colors
plt.plot(data, color='purple')  # Changed from 'red'

# Add title
plt.title('My Custom Title')
```

## Understanding the Output

### Evaluation Metrics

All examples report these metrics:

- **MSE**: Mean Squared Error (lower is better)
- **RMSE**: Root Mean Squared Error (in original units)
- **MAE**: Mean Absolute Error (average error magnitude)
- **MAPE**: Mean Absolute Percentage Error (percentage terms)

### Plots

- **Forecast plots**: Blue = training, Green = actual test, Red = forecast
- **Residuals**: Should be randomly distributed around zero
- **ACF/PACF**: Helps determine ARIMA parameters
- **Components**: Shows trend, seasonal, and residual decomposition

## Common Issues

### Import Errors

If you get import errors:
```bash
# Install missing packages
pip install -r ../requirements.txt
```

### Memory Issues (LSTM)

If LSTM runs out of memory:
```python
# Reduce batch size
history = train_model(model, X_train, y_train, batch_size=16)  # Changed from 32

# Or reduce model size
model = build_lstm_model(seq_length, units=25)  # Changed from 50
```

### Slow Training (LSTM)

If training is too slow:
```python
# Reduce epochs
history = train_model(model, X_train, y_train, epochs=20)  # Changed from 50

# Or use GPU if available (automatic with TensorFlow)
```

## Extending Examples

### Add New Method

1. Create new file: `my_method_example.py`
2. Follow this structure:
```python
"""
My Method Example

Description of the method
"""

import necessary_libraries

def load_sample_data():
    # Generate or load data
    pass

def fit_model(train_data):
    # Fit your model
    pass

def forecast(model, steps):
    # Generate forecasts
    pass

def evaluate_model(actual, predicted):
    # Calculate metrics
    pass

def main():
    # Orchestrate everything
    print("=" * 60)
    print("My Method Example")
    print("=" * 60)
    
    # Load data
    # Fit model
    # Forecast
    # Evaluate
    # Visualize
    
    print("=" * 60)
    print("Completed!")
    print("=" * 60)

if __name__ == '__main__':
    main()
```

### Combine Multiple Methods

Create a comparison script:

```python
from arima_example import fit_arima_model
from traditional_methods import triple_exponential_smoothing
# Import other methods

# Run all methods on same data
# Compare results
# Plot comparison
```

## Tips for Academic Work

### For Thesis/Skripsi

1. **Run all examples** to understand each method
2. **Choose 3-5 methods** most relevant to your data
3. **Document results** in a systematic way
4. **Compare methods** using multiple metrics
5. **Include visualizations** in your thesis

### For Journal Papers

1. **Modify examples** to match your specific problem
2. **Add statistical tests** (Diebold-Mariano, etc.)
3. **Include confidence intervals**
4. **Report computational time**
5. **Make code reproducible**

### Reproducibility Checklist

- [ ] Set random seeds
- [ ] Document Python version
- [ ] List all package versions
- [ ] Include data source/description
- [ ] Provide clear instructions
- [ ] Test on fresh environment

## Need Help?

- Check the main [README.md](../README.md)
- Read [GETTING_STARTED.md](../docs/GETTING_STARTED.md)
- Review [METHODS_COMPARISON.md](../docs/METHODS_COMPARISON.md)
- Open an issue on GitHub

---

Happy forecasting! 📊✨
