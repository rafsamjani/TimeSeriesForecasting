# Time Series Forecasting Repository

A comprehensive collection of time series forecasting methods and examples suitable for academic research, thesis work (skripsi), and journal publications.

## 📚 Overview

This repository provides implementations of various time series forecasting techniques, from traditional statistical methods to modern deep learning approaches. Each method includes:
- Complete working examples
- Detailed documentation
- Performance evaluation metrics
- Visualization utilities

## 🎯 Purpose

This repository is designed to help with:
- **Academic Research**: Ready-to-use code for thesis and dissertation work
- **Learning**: Understand different forecasting approaches
- **Comparison**: Benchmark multiple methods on your data
- **Publication**: Well-documented code suitable for reproducible research

## 📁 Repository Structure

```
TimeSeriesForecasting/
├── data/                  # Sample datasets
├── examples/              # Complete working examples
│   ├── arima_example.py
│   ├── lstm_example.py
│   ├── prophet_example.py
│   └── traditional_methods.py
├── models/                # Saved models
├── notebooks/             # Jupyter notebooks for interactive analysis
├── utils/                 # Utility functions
│   ├── data_preprocessing.py
│   └── visualization.py
├── docs/                  # Additional documentation
├── requirements.txt       # Python dependencies
└── README.md

```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/rafsamjani/TimeSeriesForecasting.git
cd TimeSeriesForecasting
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Running Examples

Each example can be run independently:

```bash
# ARIMA Example
python examples/arima_example.py

# LSTM Example
python examples/lstm_example.py

# Prophet Example
python examples/prophet_example.py

# Traditional Methods Example
python examples/traditional_methods.py
```

### Using Utility Functions

```python
from utils.data_preprocessing import load_time_series_data, train_test_split_temporal
from utils.visualization import plot_time_series, plot_forecast

# Load your data
data = load_time_series_data('your_data.csv')

# Split into train and test
train, test = train_test_split_temporal(data, test_size=0.2)

# Visualize
plot_time_series(data)
```

## 📊 Forecasting Methods

### 1. ARIMA (AutoRegressive Integrated Moving Average)

**Best for**: Stationary or trend-stationary univariate time series

**Features**:
- Statistical approach
- Handles trend and seasonality
- Interpretable parameters
- Good for short to medium-term forecasts

**Example**: `examples/arima_example.py`

**Key Parameters**:
- `p`: Order of autoregressive part
- `d`: Degree of differencing
- `q`: Order of moving average part

### 2. LSTM (Long Short-Term Memory)

**Best for**: Complex non-linear patterns, long sequences

**Features**:
- Deep learning approach
- Learns long-term dependencies
- Handles multivariate data
- Suitable for large datasets

**Example**: `examples/lstm_example.py`

**Key Parameters**:
- `seq_length`: Input sequence length
- `units`: Number of LSTM units
- `epochs`: Training iterations

### 3. Prophet

**Best for**: Business forecasting with seasonality and holidays

**Features**:
- Developed by Meta (Facebook)
- Handles missing data well
- Multiple seasonality support
- Automatic trend changepoints

**Example**: `examples/prophet_example.py`

**Key Parameters**:
- `yearly_seasonality`: Enable/disable yearly patterns
- `weekly_seasonality`: Enable/disable weekly patterns
- `changepoint_prior_scale`: Trend flexibility

### 4. Traditional Methods

**Best for**: Baseline models and simple patterns

**Includes**:
- Moving Average
- Simple Exponential Smoothing
- Double Exponential Smoothing (Holt's method)
- Triple Exponential Smoothing (Holt-Winters)
- Naive Forecast
- Seasonal Naive Forecast

**Example**: `examples/traditional_methods.py`

## 📈 Evaluation Metrics

All examples include standard evaluation metrics:

- **MSE** (Mean Squared Error): Penalizes large errors
- **RMSE** (Root Mean Squared Error): In original units
- **MAE** (Mean Absolute Error): Average absolute difference
- **MAPE** (Mean Absolute Percentage Error): Percentage error

## 🔧 Utility Modules

### Data Preprocessing (`utils/data_preprocessing.py`)

- `load_time_series_data()`: Load data from CSV
- `handle_missing_values()`: Fill or remove missing data
- `normalize_data()`: Scale data for ML models
- `create_sequences()`: Prepare data for LSTM
- `train_test_split_temporal()`: Time-aware data splitting
- `check_stationarity()`: Augmented Dickey-Fuller test
- `difference_series()`: Make series stationary

### Visualization (`utils/visualization.py`)

- `plot_time_series()`: Basic time series plot
- `plot_forecast()`: Compare actual vs predicted
- `plot_acf_pacf()`: Autocorrelation analysis
- `plot_decomposition()`: Trend, seasonal, residual
- `plot_residuals()`: Model diagnostics
- `plot_multiple_forecasts()`: Compare multiple models
- `plot_prediction_intervals()`: Uncertainty visualization

## 📝 Using for Academic Work

### For Thesis/Skripsi

1. **Choose appropriate methods** based on your data characteristics
2. **Run multiple methods** to compare performance
3. **Document results** using provided evaluation metrics
4. **Visualize findings** using utility functions
5. **Reference this repository** in your bibliography

### For Journal Publications

- Code is well-documented for reproducibility
- Evaluation metrics follow standard practices
- Visualizations are publication-ready
- Easy to extend with your own methods

### Citation

If you use this repository in your academic work, please cite:

```
@misc{timeseriesforecasting2024,
  author = {Raf Samjani},
  title = {Time Series Forecasting Repository},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/rafsamjani/TimeSeriesForecasting}
}
```

## 🎓 Learning Resources

### Recommended Reading

- **ARIMA**: "Time Series Analysis" by Box & Jenkins
- **LSTM**: "Deep Learning" by Goodfellow, Bengio, and Courville
- **Prophet**: Facebook Research paper on Prophet
- **General**: "Forecasting: Principles and Practice" by Hyndman & Athanasopoulos

### Online Courses

- Coursera: Time Series Forecasting
- DataCamp: Time Series Analysis in Python
- Fast.ai: Practical Deep Learning

## 🤝 Contributing

Contributions are welcome! Please feel free to submit:
- Bug reports
- Feature requests
- Pull requests with new methods
- Documentation improvements

## 📄 License

This project is available for academic and educational use.

## 📧 Contact

For questions or collaboration:
- GitHub: [@rafsamjani](https://github.com/rafsamjani)
- Repository: [TimeSeriesForecasting](https://github.com/rafsamjani/TimeSeriesForecasting)

## 🙏 Acknowledgments

This repository aggregates best practices from the time series forecasting community and various academic sources.

---

**Happy Forecasting! 📊📈**

*Last Updated: 2024*
