# Quick Start Guide

Get started with time series forecasting in 5 minutes!

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rafsamjani/TimeSeriesForecasting.git
cd TimeSeriesForecasting
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Your First Forecast

```python
from src.models.arima_model import ARIMAForecaster
from src.utils.helpers import generate_sample_data
from src.preprocessing.data_loader import train_test_split_ts
from src.evaluation.metrics import calculate_metrics

# Generate sample data
data = generate_sample_data(n_samples=200)

# Split into train/test
train, test = train_test_split_ts(data, test_size=0.2)

# Train model
model = ARIMAForecaster(order=(1, 1, 1))
model.fit(train)

# Make predictions
predictions = model.predict(steps=len(test))

# Evaluate
metrics = calculate_metrics(test.values, predictions.values)
print(metrics)
```

## Run Example Script

```bash
cd examples
python example_usage.py
```

## Explore Notebooks

Open Jupyter and explore the tutorial notebooks:

```bash
jupyter notebook notebooks/
```

Start with:
1. `01_data_exploration.ipynb` - Learn data analysis basics
2. `02_classical_methods.ipynb` - Try ARIMA and Prophet
3. `03_deep_learning.ipynb` - Use LSTM for forecasting

## Next Steps

- Read the full [README](README.md)
- Check [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- Run tests: `pytest tests/`

## Common Use Cases

### Sales Forecasting
```python
from src.models.prophet_model import ProphetForecaster
# Prophet works well for business data with trends and seasonality
```

### Stock Price Prediction
```python
from src.models.lstm_model import LSTMForecaster
# LSTM captures complex patterns in financial data
```

### Weather Forecasting
```python
from src.models.arima_model import ARIMAForecaster
# ARIMA is excellent for stationary weather patterns
```

## Need Help?

- Open an issue on GitHub
- Check the documentation in each module
- Review example notebooks

Happy Forecasting! 📈
