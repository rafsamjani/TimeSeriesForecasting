# Models Directory

This directory is for storing trained models that can be reused later.

## Saving Models

### ARIMA Models
```python
import pickle

# Save ARIMA model
with open('models/arima_model.pkl', 'wb') as f:
    pickle.dump(model_fit, f)

# Load ARIMA model
with open('models/arima_model.pkl', 'rb') as f:
    model_fit = pickle.load(f)
```

### LSTM Models
```python
# Save LSTM model
model.save('models/lstm_model.h5')

# Load LSTM model
from tensorflow.keras.models import load_model
model = load_model('models/lstm_model.h5')
```

### Prophet Models
```python
import pickle

# Save Prophet model
with open('models/prophet_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load Prophet model
with open('models/prophet_model.pkl', 'rb') as f:
    model = pickle.load(f)
```

### Using Joblib (Recommended for scikit-learn and similar)
```python
import joblib

# Save
joblib.dump(model, 'models/model_name.joblib')

# Load
model = joblib.load('models/model_name.joblib')
```

## Model Naming Convention

Use descriptive names that include:
- Method name
- Dataset name
- Date trained
- Version (if applicable)

Examples:
- `arima_sales_20240315_v1.pkl`
- `lstm_stock_prices_20240316.h5`
- `prophet_temperature_20240317.pkl`

## Model Metadata

Consider saving metadata alongside your model:

```python
import json

metadata = {
    'model_type': 'ARIMA',
    'parameters': {'p': 1, 'd': 1, 'q': 1},
    'training_date': '2024-03-15',
    'data_range': '2020-01-01 to 2023-12-31',
    'metrics': {
        'rmse': 5.23,
        'mae': 4.11,
        'mape': 3.45
    },
    'preprocessing': {
        'differencing': True,
        'scaling': 'minmax'
    }
}

with open('models/arima_sales_20240315_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

## Best Practices

1. **Version Control**: Use git LFS for large model files
2. **Documentation**: Always save metadata
3. **Testing**: Test loaded models before deployment
4. **Cleanup**: Remove old/unused models regularly
5. **Backup**: Keep backups of important models

## Model File Sizes

Typical sizes:
- ARIMA: <1 MB
- Exponential Smoothing: <1 MB
- Prophet: 1-10 MB
- LSTM: 1-100 MB (depends on architecture)

Note: Model files are excluded from git by default (see `.gitignore`).

## Reproducibility

To ensure reproducibility:

```python
# Set seeds
import numpy as np
import tensorflow as tf
import random

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Save environment
# pip freeze > models/requirements_model_name.txt
```

## Loading Models in Production

```python
class ModelLoader:
    """Helper class for loading models"""
    
    @staticmethod
    def load_model(model_path, model_type='arima'):
        """Load a saved model"""
        if model_type in ['arima', 'prophet']:
            import pickle
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        elif model_type == 'lstm':
            from tensorflow.keras.models import load_model
            return load_model(model_path)
        else:
            import joblib
            return joblib.load(model_path)
    
    @staticmethod
    def load_metadata(metadata_path):
        """Load model metadata"""
        import json
        with open(metadata_path, 'r') as f:
            return json.load(f)

# Usage
loader = ModelLoader()
model = loader.load_model('models/arima_model.pkl', model_type='arima')
metadata = loader.load_metadata('models/arima_model_metadata.json')
```

## Model Registry

For larger projects, consider maintaining a model registry:

```python
# models/registry.json
{
    "models": [
        {
            "name": "arima_sales_v1",
            "path": "models/arima_sales_20240315_v1.pkl",
            "type": "arima",
            "status": "production",
            "metrics": {"rmse": 5.23, "mae": 4.11}
        },
        {
            "name": "lstm_sales_v1",
            "path": "models/lstm_sales_20240316_v1.h5",
            "type": "lstm",
            "status": "testing",
            "metrics": {"rmse": 4.89, "mae": 3.87}
        }
    ]
}
```

## Security Note

⚠️ **Never load models from untrusted sources!** 
Pickle files can execute arbitrary code when loaded.

Only load models you created or from trusted sources.

---

Store your trained models here for easy reuse and deployment!
