#!/usr/bin/env python3
"""
Example workflow using REAL ENTSO-E electricity price data.

This script demonstrates how to use actual electricity market data
from the ENTSO-E Transparency Platform.
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
from data.entsoe_downloader import ENTSOEDownloader
from data.preprocessor import DataPreprocessor
from models.baseline_models import BaselineModels
from models.ml_models import MLModels
from models.time_series_models import TimeSeriesModels
from evaluation.metrics import EvaluationMetrics
from evaluation.visualization import ModelVisualization


def main():
    """Main workflow using real ENTSO-E data."""
    print("=== ELECTRICITY PRICE FORECASTING WITH REAL DATA ===\n")
    
    # Check if API token is available
    api_token = os.getenv('ENTSOE_API_TOKEN')
    if not api_token:
        print("ERROR: No ENTSO-E API token found!")
        print("Please set your API token:")
        print("export ENTSOE_API_TOKEN='your_token_here'")
        print("\nTo get a token:")
        print("1. Go to https://transparency.entsoe.eu/")
        print("2. Register for a free account")
        print("3. Get your API token")
        return
    
    print(f"Using ENTSO-E API token: {api_token[:10]}...")
    
    # 1. Download real electricity price data
    print("\n1. Downloading real electricity price data from ENTSO-E...")
    downloader = ENTSOEDownloader(api_token=api_token)
    
    # Download data for Italy (last 2 years)
    try:
        real_data = downloader.download_price_data(
            country='IT',  # Italy
            start_date='2023-01-01',
            end_date='2024-01-01',
            data_type='day_ahead',
            save_path='data/raw/italy_real_prices_2023.csv'
        )
        
        if real_data.empty:
            print("ERROR: No data downloaded. Check your API token and try again.")
            return
            
        print(f"   Downloaded {len(real_data)} real price records")
        print(f"   Date range: {real_data['datetime'].min()} to {real_data['datetime'].max()}")
        print(f"   Price range: €{real_data['price'].min():.2f} - €{real_data['price'].max():.2f}/MWh")
        
    except Exception as e:
        print(f"ERROR downloading data: {e}")
        print("Falling back to synthetic data...")
        return use_synthetic_data()
    
    # 2. Preprocess real data
    print("\n2. Preprocessing real electricity price data...")
    preprocessor = DataPreprocessor()
    
    # Set datetime as index
    real_data = real_data.set_index('datetime')
    
    # Clean the data
    clean_data = preprocessor.clean_price_data(real_data)
    print(f"   Cleaned data shape: {clean_data.shape}")
    print(f"   Missing values: {clean_data.isnull().sum().sum()}")
    
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
    print(f"   Training period: {X_train.index.min()} to {X_train.index.max()}")
    print(f"   Test period: {X_test.index.min()} to {X_test.index.max()}")
    
    # 5. Train models
    print("\n5. Training models...")
    
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
    
    # Train time series models
    ts_models = TimeSeriesModels()
    try:
        arima_model = ts_models.train_arima(y_train)
        arima_pred = ts_models.predict_arima(len(y_test))
        print("   ARIMA model trained successfully")
    except Exception as e:
        print(f"   ARIMA training failed: {e}")
        arima_pred = np.full(len(y_test), y_train.mean())
    
    ts_predictions = {'arima': arima_pred}
    ts_results = ts_models.evaluate_all(y_test, ts_predictions)
    
    # 6. Comprehensive evaluation
    print("\n6. Comprehensive model evaluation...")
    evaluator = EvaluationMetrics()
    
    # Combine all predictions
    all_predictions = {**baseline_predictions, **ml_predictions, **ts_predictions}
    all_results = {}
    
    for model_name, pred in all_predictions.items():
        metrics = evaluator.calculate_all_metrics(y_test, pred, model_name)
        all_results[model_name] = metrics
    
    # Create comparison DataFrame
    comparison_df = evaluator.compare_models(all_results)
    
    # 7. Display results
    print("\n=== REAL DATA MODEL PERFORMANCE COMPARISON ===")
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
    
    # 8. Business impact analysis
    print("\n=== BUSINESS IMPACT ANALYSIS (REAL DATA) ===")
    
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
    
    # 9. Create visualizations
    print("\n9. Creating visualizations...")
    viz = ModelVisualization()
    
    # Plot predictions vs actual
    viz.plot_predictions_vs_actual(y_test, all_predictions, 
                                 title="Real Electricity Price Forecasting Results")
    
    # Plot metrics comparison
    viz.plot_metrics_comparison(comparison_df, 
                               metrics=['rmse', 'mae', 'mape', 'directional_accuracy'],
                               title="Model Performance Comparison (Real Data)")
    
    print("\n=== REAL DATA WORKFLOW COMPLETED SUCCESSFULLY ===")
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


def use_synthetic_data():
    """Fallback to synthetic data if real data download fails."""
    print("\n=== FALLING BACK TO SYNTHETIC DATA ===")
    print("Running the synthetic data workflow...")
    
    # Import and run the synthetic workflow
    from example_workflow import main as synthetic_main
    synthetic_main()


if __name__ == "__main__":
    main()
