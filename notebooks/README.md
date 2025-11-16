# Jupyter Notebooks

This directory contains interactive Jupyter notebooks for exploring time series forecasting methods.

## Getting Started

1. Make sure Jupyter is installed:
```bash
pip install jupyter notebook
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook
```

3. Navigate to this directory and open any notebook.

## Available Notebooks

### Tutorial Notebooks
- `01_introduction_to_time_series.ipynb` - Basic concepts and visualization
- `02_data_preprocessing.ipynb` - Cleaning and preparing time series data
- `03_traditional_methods.ipynb` - Moving averages and exponential smoothing
- `04_arima_tutorial.ipynb` - ARIMA modeling step by step
- `05_lstm_tutorial.ipynb` - Deep learning for time series
- `06_prophet_tutorial.ipynb` - Using Facebook Prophet
- `07_model_comparison.ipynb` - Comparing multiple methods

### Analysis Templates
- `template_analysis.ipynb` - Template for your own analysis
- `thesis_analysis_template.ipynb` - Structured template for academic work

## Notebook Best Practices

### 1. Clear Structure
```python
# Cell 1: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cell 2: Load Data
data = pd.read_csv('data.csv')

# Cell 3: Exploratory Analysis
data.describe()

# And so on...
```

### 2. Document Your Work
Use markdown cells to explain:
- What you're doing
- Why you're doing it
- What you learned

### 3. Save Regularly
- Use `File > Save and Checkpoint`
- Consider version control with git

### 4. Export Results
```python
# Save figures
plt.savefig('results/figure1.png', dpi=300)

# Save models
import joblib
joblib.dump(model, 'models/my_model.pkl')

# Export to HTML/PDF for sharing
# File > Download as > HTML/PDF
```

## Creating Your Own Notebook

Start with this template:

```python
# %% [markdown]
# # Your Analysis Title
# 
# **Author**: Your Name
# **Date**: 2024-XX-XX
# 
# ## Objective
# Describe what you're trying to achieve

# %% [markdown]
# ## 1. Import Libraries

# %%
import sys
sys.path.append('..')  # Add parent directory to path

from utils.data_preprocessing import *
from utils.visualization import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## 2. Load and Explore Data

# %%
# Your code here

# %% [markdown]
# ## 3. Preprocessing

# %%
# Your code here

# %% [markdown]
# ## 4. Model Training

# %%
# Your code here

# %% [markdown]
# ## 5. Evaluation

# %%
# Your code here

# %% [markdown]
# ## 6. Conclusions
# 
# Summarize your findings
```

## Tips for Thesis Work

1. **One notebook per experiment** - Keep things organized
2. **Use descriptive names** - `20240315_arima_sales_forecast.ipynb`
3. **Export key results** - Save figures and tables for your thesis
4. **Add markdown explanations** - Future you will thank you
5. **Keep a master notebook** - That runs all analyses

## Converting Notebooks

### To Python Script
```bash
jupyter nbconvert --to script notebook.ipynb
```

### To HTML (for sharing)
```bash
jupyter nbconvert --to html notebook.ipynb
```

### To PDF (requires LaTeX)
```bash
jupyter nbconvert --to pdf notebook.ipynb
```

## Troubleshooting

### Kernel Issues
- Restart kernel: `Kernel > Restart`
- Clear output: `Cell > All Output > Clear`

### Import Errors
```python
# If you can't import utils
import sys
sys.path.append('..')  # Adds parent directory
```

### Large Notebooks
- Clear outputs before committing to git
- Consider splitting into multiple notebooks

## Resources

- [Jupyter Documentation](https://jupyter.org/documentation)
- [Markdown Cheatsheet](https://www.markdownguide.org/cheat-sheet/)
- [Keyboard Shortcuts](https://towardsdatascience.com/jypyter-notebook-shortcuts-bf0101a98330)

Happy analyzing! 📊✨
