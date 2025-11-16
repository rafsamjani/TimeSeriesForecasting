# Frequently Asked Questions (FAQ)

## General Questions

### Q: What is this repository for?
**A:** This repository provides a comprehensive collection of time series forecasting methods and examples, specifically designed for academic work like thesis (skripsi), research papers, and learning purposes.

### Q: Do I need to be a Python expert to use this?
**A:** No! The examples are well-documented and beginner-friendly. Basic Python knowledge is helpful but not required to get started.

### Q: Can I use this for my thesis/skripsi?
**A:** Absolutely! That's one of the main purposes. All code is well-documented, includes evaluation metrics, and follows academic best practices.

### Q: Is this repository suitable for commercial projects?
**A:** Yes, you can use it for both academic and commercial purposes. However, always verify the licenses of the underlying libraries.

---

## Installation & Setup

### Q: What Python version do I need?
**A:** Python 3.8 or higher is recommended. The code has been tested on Python 3.8+.

### Q: How do I install all dependencies?
**A:**
```bash
pip install -r requirements.txt
```

### Q: The installation is taking too long. Why?
**A:** Some packages like TensorFlow and Prophet are large. Be patient, or install only what you need:
```bash
# For ARIMA and traditional methods only
pip install numpy pandas matplotlib statsmodels scikit-learn

# Add LSTM support
pip install tensorflow

# Add Prophet support
pip install prophet
```

### Q: I'm getting import errors. What should I do?
**A:** Make sure you've installed all dependencies and you're in the correct directory. If using Jupyter notebooks, restart the kernel after installing packages.

---

## Data Questions

### Q: Can I use my own data?
**A:** Yes! Replace the `load_sample_data()` function in any example with code to load your data. Your data should be in a pandas DataFrame with a datetime index.

### Q: What format should my data be in?
**A:** CSV format with at least two columns: date and value(s). Example:
```
date,value
2020-01-01,100
2020-01-02,105
2020-01-03,103
```

### Q: How much data do I need?
**A:**
- **Minimum**: 30-50 observations for traditional methods and ARIMA
- **Recommended**: 100+ observations for most methods
- **For LSTM**: 1000+ observations recommended

### Q: My data has missing values. What should I do?
**A:** Use the `handle_missing_values()` function from `utils.data_preprocessing`:
```python
from utils.data_preprocessing import handle_missing_values
data = handle_missing_values(data, method='interpolate')
```

### Q: How do I check if my data is stationary?
**A:**
```python
from utils.data_preprocessing import check_stationarity
result = check_stationarity(data, column='value')
print(result)
```

---

## Method Selection

### Q: Which method should I use?
**A:** It depends on your data and requirements:
- **Small dataset (<100 points)**: Holt-Winters, ARIMA
- **Strong seasonality**: Prophet, Holt-Winters
- **Large dataset (>1000 points)**: LSTM, any method
- **Need interpretability**: ARIMA, traditional methods
- **For thesis**: Compare 3-5 methods (e.g., ARIMA + Prophet + LSTM)

See [METHODS_COMPARISON.md](METHODS_COMPARISON.md) for detailed guidance.

### Q: Can I use multiple methods together?
**A:** Yes! Ensemble methods often work well:
```python
ensemble_forecast = 0.4 * arima + 0.3 * prophet + 0.3 * lstm
```

### Q: What's the difference between ARIMA and SARIMA?
**A:** SARIMA extends ARIMA with seasonal components. Use SARIMA when you have clear seasonal patterns (e.g., monthly, quarterly).

---

## Model Training

### Q: How long does training take?
**A:**
- **ARIMA**: Seconds to minutes
- **Prophet**: Seconds to minutes
- **Holt-Winters**: Seconds
- **LSTM**: Minutes to hours (depending on data size and hardware)

### Q: My LSTM is training very slowly. What can I do?
**A:**
1. Reduce the number of epochs
2. Reduce batch size
3. Reduce model complexity (fewer units/layers)
4. Use a GPU if available
5. Use a smaller subset of data for testing

### Q: How do I know if my model is overfitting?
**A:** Check if training performance is much better than test performance. Use the validation split during training:
```python
history = model.fit(X_train, y_train, validation_split=0.1)
# Compare training and validation loss
```

### Q: Can I save my trained models?
**A:** Yes! See [models/README.md](../models/README.md) for details:
```python
# For ARIMA/Prophet
import pickle
with open('models/my_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# For LSTM
model.save('models/my_model.h5')
```

---

## Evaluation & Results

### Q: Which metric should I report in my thesis?
**A:** Report multiple metrics:
- **RMSE**: Most common, penalizes large errors
- **MAE**: Easier to interpret
- **MAPE**: Scale-independent, good for comparison

Always report what's standard in your field.

### Q: What's a good RMSE value?
**A:** It depends on your data scale. An RMSE of 5 is excellent if your values range from 0-100, but poor if they range from 0-10000. Focus on relative comparison between methods.

### Q: How do I know if my forecast is good?
**A:**
1. Compare against baseline (naive forecast)
2. Check if residuals are random (no pattern)
3. Compare with other methods
4. Validate on out-of-sample data
5. Check business/domain relevance

### Q: Should I compare my model to a baseline?
**A:** Yes! Always compare to simple baselines like:
- Naive forecast (last observation)
- Moving average
- Seasonal naive

This shows the value of your more complex model.

---

## Academic Use

### Q: How do I cite this repository in my thesis?
**A:**
```
@misc{timeseriesforecasting2024,
  author = {Raf Samjani},
  title = {Time Series Forecasting Repository},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/rafsamjani/TimeSeriesForecasting}
}
```

### Q: Can I modify the code for my research?
**A:** Yes! The code is designed to be extended and modified. Feel free to adapt it to your needs.

### Q: What should I include in my thesis methodology section?
**A:**
1. Description of each method used
2. Why you chose these methods
3. Parameter values and how you selected them
4. Evaluation metrics used
5. Hardware/software specifications
6. Links to code/data (if allowed)

### Q: How many methods should I compare in my thesis?
**A:** Typically 3-5 methods:
- 1-2 traditional/baseline methods
- 2-3 advanced methods
- At least one statistical and one machine learning approach

---

## Troubleshooting

### Q: I get "No module named 'prophet'" error
**A:** Install Prophet separately:
```bash
pip install prophet
```

### Q: I get "No module named 'tensorflow'" error
**A:** Install TensorFlow:
```bash
pip install tensorflow
```

### Q: The examples generate plot files. Where are they?
**A:** Plot files are saved in the directory where you run the script. They're named like `arima_forecast.png`, `lstm_forecast.png`, etc.

### Q: Can I run examples in Jupyter notebook?
**A:** Yes! Either:
1. Use the provided notebook in `notebooks/`
2. Copy code from examples into notebook cells

### Q: I'm getting a "convergence warning" with ARIMA
**A:** This is common. Try:
1. Different ARIMA parameters (p, d, q)
2. Differencing your data first
3. Increasing max iterations

### Q: My Prophet model is too wiggly/smooth
**A:** Adjust `changepoint_prior_scale`:
```python
# More flexible (wiggly)
model = Prophet(changepoint_prior_scale=0.5)

# Less flexible (smooth)
model = Prophet(changepoint_prior_scale=0.001)
```

---

## Visualization

### Q: Can I customize the plots?
**A:** Yes! All plotting code is exposed. Modify colors, sizes, labels, etc. in the example files.

### Q: How do I save plots in different formats?
**A:**
```python
# PNG (default)
plt.savefig('plot.png', dpi=300)

# PDF (for papers)
plt.savefig('plot.pdf')

# SVG (vector graphics)
plt.savefig('plot.svg')
```

### Q: Can I make interactive plots?
**A:** Yes! Install plotly:
```bash
pip install plotly
```
Then use Plotly for interactive visualizations.

---

## Performance

### Q: How can I make predictions faster?
**A:**
1. Use simpler models for quick prototyping
2. Cache trained models (save/load)
3. Reduce data size for testing
4. Use vectorized operations

### Q: Can I use GPU for LSTM?
**A:** Yes! TensorFlow automatically uses GPU if available. Install:
```bash
pip install tensorflow-gpu
```

### Q: My computer is running out of memory
**A:**
1. Reduce batch size
2. Use smaller sequences
3. Train on subset of data
4. Use simpler model architecture

---

## Advanced Topics

### Q: How do I implement cross-validation for time series?
**A:** Use time series cross-validation (walk-forward validation):
```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(data):
    train, test = data[train_idx], data[test_idx]
    # Train and evaluate
```

### Q: Can I forecast multiple steps ahead?
**A:** Yes! All methods support multi-step forecasting:
```python
# ARIMA
forecast = model.forecast(steps=30)  # 30 steps ahead

# LSTM
forecast = forecast_lstm(model, data, seq_length, scaler, steps=30)

# Prophet
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
```

### Q: How do I handle multivariate time series?
**A:** LSTM supports multivariate data naturally. For ARIMA, use VAR (Vector Autoregression) from statsmodels.

### Q: Can I add external regressors (exogenous variables)?
**A:** Yes!
- **ARIMA**: Use ARIMAX or SARIMAX from statsmodels
- **Prophet**: Use `add_regressor()` method
- **LSTM**: Include additional features in input

---

## Still Have Questions?

1. Check the [Getting Started Guide](GETTING_STARTED.md)
2. Review [Methods Comparison](METHODS_COMPARISON.md)
3. Look at example code in `examples/`
4. Open an issue on GitHub
5. Read the documentation of specific libraries (statsmodels, TensorFlow, Prophet)

---

*This FAQ is regularly updated based on common questions. If your question isn't here, please ask!*
