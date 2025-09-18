#!/usr/bin/env python3
"""
Example workflow for electricity price forecasting.

This script demonstrates the complete workflow from data generation
to model evaluation and business impact analysis.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data.preprocessor import DataPreprocessor
from models.baseline_models import BaselineModels
from models.ml_models import MLModels
from models.time_series_models import TimeSeriesModels
from evaluation.metrics import EvaluationMetrics
from evaluation.visualization import ModelVisualization


def generate_synthetic_data(n_samples=8760, start_date='2023-01-01'):
    """Generate synthetic electricity price data."""
    np.random.seed(42)
    
    # Create time index
    dates = pd.date_range(start_date, periods=n_samples, freq='H')
    
    # Base price level
    base_price = 50
    
    # Daily seasonality (higher prices during day, lower at night)
    daily_pattern = 15 * np.sin(2 * np.pi * np.arange(n_samples) / 24 - np.pi/2)
    
    # Weekly seasonality (higher prices on weekdays)
    weekly_pattern = 5 * (dates.dayofweek < 5).astype(int)
    
    # Monthly seasonality (higher prices in winter and summer)
    monthly_pattern = 10 * np.sin(2 * np.pi * dates.month / 12)
    
    # Trend (slight upward trend over time)
    trend = 0.001 * np.arange(n_samples)
    
    # Random noise
    noise = np.random.normal(0, 3, n_samples)
    
    # Price spikes (occasional high prices)
    spike_probability = 0.02  # 2% chance of spike per hour
    spikes = np.random.binomial(1, spike_probability, n_samples) * np.random.exponential(20, n_samples)
    
    # Combine all components
    price = base_price + daily_pattern + weekly_pattern + monthly_pattern + trend + noise + spikes
    
    # Ensure non-negative prices
    price = np.maximum(price, 0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'datetime': dates,
        'price': price,
        'hour': dates.hour,
        'day_of_week': dates.dayofweek,
        'month': dates.month,
        'is_weekend': (dates.dayofweek >= 5).astype(int)
    })
    
    return df


def main():
    """Main workflow function."""
    print("=== ELECTRICITY PRICE FORECASTING WORKFLOW ===\n")
    
    # 1. Generate synthetic data
    print("1. Generating synthetic electricity price data...")
    data = generate_synthetic_data(n_samples=8760, start_date='2023-01-01')
    print(f"   Generated {len(data)} hourly records")
    print(f"   Price range: €{data['price'].min():.2f} - €{data['price'].max():.2f}/MWh")
    
    # 2. Preprocess data
    print("\n2. Preprocessing data...")
    preprocessor = DataPreprocessor()
    data = data.set_index('datetime')
    clean_data = preprocessor.clean_price_data(data)
    print(f"   Cleaned data shape: {clean_data.shape}")
    
    # 3. Engineer features
    print("\n3. Engineering features...")
    features_df = preprocessor.engineer_features(clean_data)
    print(f"   Features shape: {features_df.shape}")
    print(f"   Number of features: {len(features_df.columns)}")
    
    # 4. Prepare training data
    print("\n4. Preparing training and test data...")
    X_train, X_test, y_train, y_test = preprocessor.prepare_training_data(
        target_column='price',
        test_size=0.2
    )
    print(f"   Training data: {X_train.shape}")
    print(f"   Test data: {X_test.shape}")
    
    # 5. Train baseline models
    print("\n5. Training baseline models...")
    baseline_models = BaselineModels()
    baseline_models.train_all(X_train, y_train)
    baseline_predictions = baseline_models.predict_all(X_test)
    baseline_results = baseline_models.evaluate_all(y_test, baseline_predictions)
    print("   Baseline models trained successfully")
    
    # 6. Train ML models
    print("\n6. Training machine learning models...")
    ml_models = MLModels()
    ml_models.train_all(X_train, y_train, tune_hyperparameters=False)
    ml_predictions = ml_models.predict_all(X_test)
    ml_results = ml_models.evaluate_all(y_test, ml_predictions)
    print("   ML models trained successfully")
    
    # 7. Train time series models
    print("\n7. Training time series models...")
    ts_models = TimeSeriesModels()
    
    # Train ARIMA
    try:
        arima_model = ts_models.train_arima(y_train)
        arima_pred = ts_models.predict_arima(len(y_test))
        print("   ARIMA model trained successfully")
    except Exception as e:
        print(f"   ARIMA training failed: {e}")
        arima_pred = np.full(len(y_test), y_train.mean())
    
    # Train SARIMAX
    try:
        exog_cols = ['hour', 'day_of_week', 'is_weekend']
        exog_train = X_train[exog_cols] if all(col in X_train.columns for col in exog_cols) else None
        exog_test = X_test[exog_cols] if exog_train is not None else None
        
        sarimax_model = ts_models.train_sarimax(y_train, exog_train)
        sarimax_pred = ts_models.predict_sarimax(len(y_test), exog_test)
        print("   SARIMAX model trained successfully")
    except Exception as e:
        print(f"   SARIMAX training failed: {e}")
        sarimax_pred = np.full(len(y_test), y_train.mean())
    
    ts_predictions = {
        'arima': arima_pred,
        'sarimax': sarimax_pred
    }
    ts_results = ts_models.evaluate_all(y_test, ts_predictions)
    
    # 8. Comprehensive evaluation
    print("\n8. Comprehensive model evaluation...")
    evaluator = EvaluationMetrics()
    
    # Combine all predictions
    all_predictions = {**baseline_predictions, **ml_predictions, **ts_predictions}
    all_results = {}
    
    for model_name, pred in all_predictions.items():
        metrics = evaluator.calculate_all_metrics(y_test, pred, model_name)
        all_results[model_name] = metrics
    
    # Create comparison DataFrame
    comparison_df = evaluator.compare_models(all_results)
    
    # 9. Display results
    print("\n=== MODEL PERFORMANCE COMPARISON ===")
    print(comparison_df.round(4))
    
    # Get best model
    best_model_name = comparison_df.index[0]
    best_rmse = comparison_df.loc[best_model_name, 'rmse']
    best_mae = comparison_df.loc[best_model_name, 'mae']
    best_mape = comparison_df.loc[best_model_name, 'mape']
    
    print(f"\n=== BEST MODEL: {best_model_name.upper()} ===")
    print(f"RMSE: {best_rmse:.4f} €/MWh")
    print(f"MAE: {best_mae:.4f} €/MWh")
    print(f"MAPE: {best_mape:.4f}%")
    
    # 10. Business impact analysis
    print("\n=== BUSINESS IMPACT ANALYSIS ===")
    
    # Cost estimation accuracy
    best_pred = all_predictions[best_model_name]
    total_cost_error = np.sum(np.abs(y_test - best_pred))
    avg_cost_error = total_cost_error / len(y_test)
    
    print(f"Total cost error (1 MWh): €{total_cost_error:.2f}")
    print(f"Average cost error per hour: €{avg_cost_error:.2f}")
    
    # Peak price accuracy
    price_75th = np.percentile(y_test, 75)
    peak_mask = y_test >= price_75th
    peak_prices = y_test[peak_mask]
    peak_pred = best_pred[peak_mask]
    
    if len(peak_prices) > 0:
        peak_mae = np.mean(np.abs(peak_prices - peak_pred))
        peak_mape = np.mean(np.abs((peak_prices - peak_pred) / peak_prices)) * 100
        print(f"Peak price accuracy (>{price_75th:.1f} €/MWh): MAE={peak_mae:.2f}, MAPE={peak_mape:.1f}%")
    
    # Directional accuracy
    y_diff = np.diff(y_test.values)
    pred_diff = np.diff(best_pred)
    directional_acc = np.mean((y_diff * pred_diff) > 0) * 100
    print(f"Directional accuracy: {directional_acc:.1f}%")
    
    # 11. Create visualizations
    print("\n11. Creating visualizations...")
    viz = ModelVisualization()
    
    # Plot predictions vs actual
    viz.plot_predictions_vs_actual(y_test, all_predictions, 
                                 title="Electricity Price Forecasting Results")
    
    # Plot metrics comparison
    viz.plot_metrics_comparison(comparison_df, 
                               metrics=['rmse', 'mae', 'mape', 'directional_accuracy'],
                               title="Model Performance Comparison")
    
    # Create comprehensive dashboard
    viz.create_dashboard(y_test, all_predictions, comparison_df,
                        title="Electricity Price Forecasting - Complete Analysis")
    
    print("\n=== WORKFLOW COMPLETED SUCCESSFULLY ===")
    print(f"Best model: {best_model_name}")
    print(f"Expected RMSE: {best_rmse:.2f} €/MWh")
    print(f"Expected MAE: {best_mae:.2f} €/MWh")
    print(f"Directional accuracy: {directional_acc:.1f}%")
    
    if best_rmse < 5:
        print("Model accuracy is excellent for operational planning")
    elif best_rmse < 10:
        print("Model accuracy is good for operational planning")
    else:
        print("Model accuracy may need improvement for operational planning")


if __name__ == "__main__":
    main()
