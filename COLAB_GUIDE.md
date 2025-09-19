# Google Colab Setup Guide

## Quick Start

### 1. Open Google Colab
- Go to https://colab.research.google.com/
- Create a new notebook

### 2. Setup Cell (Run First)
```python
# Install packages and clone repository
!pip install xgboost lightgbm prophet tensorflow torch plotly streamlit
!git clone https://github.com/tommasomalaguti/energy_price_predictor.git
%cd energy_price_predictor

print("Setup complete!")
```

### 3. Import Libraries
```python
import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data.entsoe_downloader import ENTSOEDownloader
from data.preprocessor import DataPreprocessor
from models.baseline_models import BaselineModels
from models.ml_models import MLModels
from models.time_series_models import TimeSeriesModels
from evaluation.metrics import EvaluationMetrics
from evaluation.visualization import ModelVisualization

print("Libraries imported successfully!")
```

### 4. Data Collection (Choose One)

#### Option A: Real Data (Requires API Token)
```python
# Set your ENTSO-E API token
ENTSOE_API_TOKEN = "your_token_here"  # Replace with your actual token

if ENTSOE_API_TOKEN != "your_token_here":
    downloader = ENTSOEDownloader(api_token=ENTSOE_API_TOKEN)
    data = downloader.download_price_data(
        country='IT',
        start_date='2023-01-01',
        end_date='2024-01-01',
        data_type='day_ahead'
    )
    print(f"Downloaded {len(data)} real price records")
else:
    print("Please set your ENTSO-E API token to download real data")
```

#### Option B: Synthetic Data (No API Required)
```python
# Generate synthetic data
def generate_synthetic_data(n_samples=8760, start_date='2023-01-01'):
    np.random.seed(42)
    dates = pd.date_range(start_date, periods=n_samples, freq='H')
    
    base_price = 50
    daily_pattern = 15 * np.sin(2 * np.pi * np.arange(n_samples) / 24 - np.pi/2)
    weekly_pattern = 5 * (dates.dayofweek < 5).astype(int)
    monthly_pattern = 10 * np.sin(2 * np.pi * dates.month / 12)
    trend = 0.001 * np.arange(n_samples)
    noise = np.random.normal(0, 3, n_samples)
    spikes = np.random.binomial(1, 0.02, n_samples) * np.random.exponential(20, n_samples)
    
    price = base_price + daily_pattern + weekly_pattern + monthly_pattern + trend + noise + spikes
    price = np.maximum(price, 0)
    
    return pd.DataFrame({
        'datetime': dates,
        'price': price,
        'hour': dates.hour,
        'day_of_week': dates.dayofweek,
        'month': dates.month,
        'is_weekend': (dates.dayofweek >= 5).astype(int)
    })

data = generate_synthetic_data()
print(f"Generated {len(data)} synthetic price records")
```

### 5. Data Preprocessing
```python
# Preprocess data
preprocessor = DataPreprocessor()
data = data.set_index('datetime')
clean_data = preprocessor.clean_price_data(data)
features_df = preprocessor.engineer_features(clean_data)

# Handle missing values (important for ML models)
print("Handling missing values...")
print(f"Missing values before: {features_df.isnull().sum().sum()}")

# Fill missing values with forward fill, then backward fill
features_df = features_df.fillna(method='ffill').fillna(method='bfill')

# If still missing values, fill with mean
features_df = features_df.fillna(features_df.mean())

print(f"Missing values after: {features_df.isnull().sum().sum()}")

# Prepare training data
X_train, X_test, y_train, y_test = preprocessor.prepare_training_data(
    target_column='price',
    test_size=0.2
)

print(f"Training data: {X_train.shape}")
print(f"Test data: {X_test.shape}")
print(f"Training missing values: {X_train.isnull().sum().sum()}")
print(f"Test missing values: {X_test.isnull().sum().sum()}")
```

### 6. Train Models
```python
# Train baseline models
baseline_models = BaselineModels()
baseline_models.train_all(X_train, y_train)
baseline_predictions = baseline_models.predict_all(X_test)
baseline_results = baseline_models.evaluate_all(y_test, baseline_predictions)

# Train ML models
ml_models = MLModels()
ml_models.train_all(X_train, y_train, tune_hyperparameters=False)
ml_predictions = ml_models.predict_all(X_test)
ml_results = ml_models.evaluate_all(y_test, ml_predictions)

print("Models trained successfully!")
```

### 6.1 Quick Fix for NaN Values (if you get errors)
```python
# If you get NaN errors, run this cell to fix the data
print("Fixing NaN values in training data...")
print(f"X_train NaN count: {X_train.isnull().sum().sum()}")
print(f"X_test NaN count: {X_test.isnull().sum().sum()}")

# Fill missing values
X_train = X_train.fillna(method='ffill').fillna(method='bfill').fillna(X_train.mean())
X_test = X_test.fillna(method='ffill').fillna(method='bfill').fillna(X_train.mean())

print(f"After fixing - X_train NaN count: {X_train.isnull().sum().sum()}")
print(f"After fixing - X_test NaN count: {X_test.isnull().sum().sum()}")

# Now retrain the models
print("Retraining models with clean data...")
ml_models = MLModels()
ml_models.train_all(X_train, y_train, tune_hyperparameters=False)
ml_predictions = ml_models.predict_all(X_test)
ml_results = ml_models.evaluate_all(y_test, ml_predictions)

print("Models retrained successfully!")
```

### 7. Evaluate and Visualize
```python
# Comprehensive evaluation
evaluator = EvaluationMetrics()
all_predictions = {**baseline_predictions, **ml_predictions}
all_results = {}

for model_name, pred in all_predictions.items():
    metrics = evaluator.calculate_all_metrics(y_test, pred, model_name)
    all_results[model_name] = metrics

comparison_df = evaluator.compare_models(all_results)
print("Model Performance Comparison:")
print(comparison_df.round(4))

# Create interactive plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test.values, 
                mode='lines', name='Actual', line=dict(width=2, color='black')))

for model_name, pred in all_predictions.items():
    fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=pred, 
                            mode='lines', name=model_name))

fig.update_layout(title='Electricity Price Predictions vs Actual',
                  xaxis_title='Time Index', yaxis_title='Price (€/MWh)')
fig.show()
```

### 8. Business Analysis
```python
# Business impact analysis
best_model_name = comparison_df.index[0]
best_pred = all_predictions[best_model_name]

total_cost_error = np.sum(np.abs(y_test - best_pred))
avg_cost_error = total_cost_error / len(y_test)

print(f"Best Model: {best_model_name}")
print(f"RMSE: {comparison_df.loc[best_model_name, 'rmse']:.2f} €/MWh")
print(f"MAE: {comparison_df.loc[best_model_name, 'mae']:.2f} €/MWh")
print(f"Total cost error: €{total_cost_error:.2f}")
print(f"Average cost error per hour: €{avg_cost_error:.2f}")
```

## Colab Advantages

1. **Free GPU Access**: Enable GPU in Runtime > Change runtime type
2. **Pre-installed Libraries**: Most ML libraries already available
3. **Interactive Visualizations**: Plotly works great in Colab
4. **Easy Sharing**: Share notebooks with others
5. **Cloud Storage**: Save results to Google Drive

## Tips for Colab

1. **Enable GPU**: Runtime > Change runtime type > GPU
2. **Save Progress**: Mount Google Drive to save results
3. **Session Timeout**: Colab sessions timeout after inactivity
4. **Memory Limits**: Large datasets may hit memory limits
5. **Install Once**: Run setup cell only once per session

## Next Steps

1. **Get API Token**: Register at https://transparency.entsoe.eu/
2. **Experiment**: Try different models and parameters
3. **Add Features**: Include weather data, demand forecasts
4. **Deploy**: Export models for production use
5. **Share**: Share your Colab notebook with others

## Troubleshooting

- **Import Errors**: Make sure you're in the correct directory
- **Memory Issues**: Reduce dataset size or use smaller models
- **Timeout**: Save work frequently and restart if needed
- **API Errors**: Check your ENTSO-E token is valid
