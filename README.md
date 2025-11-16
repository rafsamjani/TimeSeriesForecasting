# Time Series Forecasting Repository

A comprehensive collection of time series forecasting methods and models suitable for academic research, thesis work, and journal publications.

## 📋 Overview

This repository contains implementations of various time series forecasting algorithms, from classical statistical methods to modern deep learning approaches. It is designed to help students, researchers, and practitioners explore and apply different forecasting techniques.

## 🎯 Features

- **Classical Methods**: ARIMA, SARIMA, Exponential Smoothing
- **Machine Learning**: Random Forest, XGBoost for time series
- **Deep Learning**: LSTM, GRU, CNN-LSTM models
- **Modern Approaches**: Facebook Prophet, Neural Prophet
- **Preprocessing Tools**: Stationarity tests, normalization, feature engineering
- **Evaluation Metrics**: MAE, RMSE, MAPE, R², and more
- **Visualization**: Interactive plots for analysis and results
- **Jupyter Notebooks**: Step-by-step tutorials and examples

## 📁 Project Structure

```
TimeSeriesForecasting/
├── data/
│   ├── raw/              # Raw datasets
│   └── processed/        # Processed datasets
├── models/
│   └── saved/            # Saved model files
├── src/
│   ├── models/           # Model implementations
│   ├── preprocessing/    # Data preprocessing utilities
│   ├── evaluation/       # Evaluation metrics
│   ├── visualization/    # Plotting utilities
│   └── utils/            # Helper functions
├── notebooks/            # Jupyter notebooks with examples
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🚀 Installation

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

## 📚 Usage

### Quick Start

```python
from src.models.arima_model import ARIMAForecaster
from src.preprocessing.data_loader import load_time_series
from src.evaluation.metrics import calculate_metrics

# Load data
data = load_time_series('data/raw/your_data.csv')

# Train model
model = ARIMAForecaster(order=(1, 1, 1))
model.fit(data['train'])

# Make predictions
predictions = model.predict(steps=30)

# Evaluate
metrics = calculate_metrics(data['test'], predictions)
print(metrics)
```

### Available Models

1. **ARIMA/SARIMA**: For univariate time series with trends and seasonality
2. **Prophet**: Facebook's forecasting tool for business time series
3. **LSTM**: Deep learning for complex patterns and long-term dependencies
4. **XGBoost**: Gradient boosting for time series regression

### Jupyter Notebooks

Explore the `notebooks/` directory for detailed tutorials:
- `01_data_exploration.ipynb`: Data loading and exploratory analysis
- `02_classical_methods.ipynb`: ARIMA and exponential smoothing
- `03_machine_learning.ipynb`: ML-based forecasting
- `04_deep_learning.ipynb`: LSTM and neural networks
- `05_model_comparison.ipynb`: Compare different approaches

## 📊 Example Results

The repository includes examples with various datasets:
- Sales forecasting
- Stock price prediction
- Energy consumption forecasting
- Weather data prediction

## 🧪 Testing

Run tests using pytest:
```bash
pytest tests/
```

## 📖 Documentation

Each module contains detailed documentation:
- Model APIs and parameters
- Preprocessing functions
- Evaluation metrics explanations
- Best practices and tips

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact the repository maintainer.

## 📚 Citation

If you use this repository for your research or thesis, please cite:

```bibtex
@misc{timeseriesforecasting2025,
  title={Time Series Forecasting Repository},
  author={TimeSeriesForecasting Contributors},
  year={2025},
  publisher={GitHub},
  url={https://github.com/rafsamjani/TimeSeriesForecasting}
}
```

## 🌟 Acknowledgments

This repository builds upon various open-source libraries and research papers in time series forecasting.

## 📝 TODO

- [ ] Add more advanced models (Transformers, N-BEATS)
- [ ] Include multivariate time series examples
- [ ] Add automated model selection
- [ ] Create web dashboard for visualization
- [ ] Add more real-world datasets
