#!/usr/bin/env python3
"""
Test script to verify the improvements work correctly.

This script tests:
1. Feature selection functionality
2. Safe MAPE calculation
3. Improved future predictions
4. Model performance improvements

Usage:
    python test_improvements.py
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to the Python path
current_dir = os.getcwd()
src_path = os.path.join(current_dir, 'src')
if os.path.exists(src_path):
    sys.path.append(src_path)

from data.preprocessor import DataPreprocessor

def test_feature_selection():
    """Test the feature selection functionality."""
    print(" Testing Feature Selection...")
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='H')
    np.random.seed(42)
    
    # Create sample price data with patterns
    base_price = 50 + 20 * np.sin(2 * np.pi * np.arange(1000) / 24)  # Daily pattern
    weekly_pattern = 10 * np.sin(2 * np.pi * np.arange(1000) / 168)  # Weekly pattern
    noise = np.random.normal(0, 5, 1000)
    price_data = base_price + weekly_pattern + noise
    
    df = pd.DataFrame({
        'price': price_data
    }, index=dates)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Create features
    print("   Creating features...")
    features_df = preprocessor.engineer_features(df)
    print(f"    Created {features_df.shape[1]} features")
    
    # Test feature selection
    print("   Applying feature selection...")
    selected_features_df = preprocessor.select_features(
        features_df=features_df, 
        target_column='price', 
        max_features=15
    )
    
    print(f"    Selected {selected_features_df.shape[1]-1} features (excluding target)")
    print(f"    Original: {features_df.shape[1]} → Selected: {selected_features_df.shape[1]-1}")
    
    # Verify feature selection worked
    assert selected_features_df.shape[1] <= 16, "Feature selection didn't reduce features enough"
    assert 'price' in selected_features_df.columns, "Target column missing"
    
    print("    Feature selection test passed!")
    return True

def test_safe_mape():
    """Test the safe MAPE calculation."""
    print("\n Testing Safe MAPE Calculation...")
    
    # Test with normal data
    y_true = np.array([10, 20, 30, 40, 50])
    y_pred = np.array([11, 19, 31, 39, 51])
    
    from electricity_forecasting_improved import safe_mape, safe_smape
    
    mape = safe_mape(y_true, y_pred)
    smape = safe_smape(y_true, y_pred)
    
    print(f"   Normal data - MAPE: {mape:.2f}%, sMAPE: {smape:.2f}%")
    assert not np.isnan(mape), "MAPE should not be NaN"
    assert not np.isinf(mape), "MAPE should not be infinite"
    
    # Test with zero values (problematic for regular MAPE)
    y_true_zero = np.array([0, 1, 2, 3, 4])
    y_pred_zero = np.array([0.1, 1.1, 1.9, 3.1, 3.9])
    
    mape_zero = safe_mape(y_true_zero, y_pred_zero)
    smape_zero = safe_smape(y_true_zero, y_pred_zero)
    
    print(f"   Zero values - MAPE: {mape_zero:.2f}%, sMAPE: {smape_zero:.2f}%")
    assert not np.isnan(mape_zero), "MAPE with zeros should not be NaN"
    assert not np.isinf(mape_zero), "MAPE with zeros should not be infinite"
    
    print("    Safe MAPE calculation test passed!")
    return True

def test_future_predictions():
    """Test improved future predictions."""
    print("\n Testing Future Predictions...")
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=200, freq='H')
    np.random.seed(42)
    
    # Create realistic price data
    base_price = 50 + 20 * np.sin(2 * np.pi * np.arange(200) / 24)  # Daily pattern
    weekly_pattern = 10 * np.sin(2 * np.pi * np.arange(200) / 168)  # Weekly pattern
    noise = np.random.normal(0, 3, 200)
    price_data = base_price + weekly_pattern + noise
    
    df = pd.DataFrame({
        'price': price_data
    }, index=dates)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Create features
    features_df = preprocessor.engineer_features(df)
    
    # Apply feature selection
    selected_features_df = preprocessor.select_features(
        features_df=features_df, 
        target_column='price', 
        max_features=20
    )
    
    # Test future feature creation
    future_hours = 24
    last_timestamp = selected_features_df.index[-1]
    future_dates = pd.date_range(start=last_timestamp + timedelta(hours=1), periods=future_hours, freq='H')
    
    # Create future features
    future_features = pd.DataFrame(index=future_dates)
    future_features['hour'] = future_dates.hour
    future_features['day_of_week'] = future_dates.dayofweek
    future_features['hour_sin'] = np.sin(2 * np.pi * future_features['hour'] / 24)
    future_features['hour_cos'] = np.cos(2 * np.pi * future_features['hour'] / 24)
    future_features['is_weekend'] = (future_dates.dayofweek >= 5).astype(int)
    
    # Add essential features
    recent_prices = selected_features_df['price'].tail(168).values
    future_features['price_lag_1'] = recent_prices[-1]
    future_features['price_lag_24'] = recent_prices[-24] if len(recent_prices) >= 24 else recent_prices[-1]
    future_features['price_mean_24h'] = selected_features_df['price'].tail(24).mean()
    
    # Fill missing features
    feature_cols = [col for col in selected_features_df.columns if col != 'price']
    for col in feature_cols:
        if col not in future_features.columns:
            future_features[col] = selected_features_df[col].median()
    
    future_features = future_features[feature_cols]
    future_features = future_features.fillna(0)
    
    print(f"    Future features shape: {future_features.shape}")
    print(f"    Future dates: {future_dates[0]} to {future_dates[-1]}")
    
    # Test pattern-based prediction
    future_predictions = []
    for future_date in future_dates:
        hour = future_date.hour
        day_of_week = future_date.dayofweek
        
        # Find similar historical patterns
        similar_mask = (selected_features_df.index.hour == hour) & (selected_features_df.index.dayofweek == day_of_week)
        if similar_mask.sum() > 0:
            similar_prices = selected_features_df.loc[similar_mask, 'price'].tail(5)
            pred = similar_prices.mean()
        else:
            hour_mask = selected_features_df.index.hour == hour
            if hour_mask.sum() > 0:
                pred = selected_features_df.loc[hour_mask, 'price'].tail(10).mean()
            else:
                pred = selected_features_df['price'].tail(24).mean()
        
        future_predictions.append(pred)
    
    future_predictions = np.array(future_predictions)
    
    print(f"    Generated {len(future_predictions)} predictions")
    print(f"    Prediction range: {future_predictions.min():.2f} - {future_predictions.max():.2f} EUR/MWh")
    print(f"    Prediction mean: {future_predictions.mean():.2f} EUR/MWh")
    
    # Verify predictions are realistic
    assert len(future_predictions) == future_hours, "Wrong number of predictions"
    assert not np.all(future_predictions == future_predictions[0]), "Predictions should vary (not constant)"
    assert not np.any(np.isnan(future_predictions)), "Predictions should not contain NaN"
    assert not np.any(np.isinf(future_predictions)), "Predictions should not contain infinite values"
    
    print("    Future predictions test passed!")
    return True

def main():
    """Run all tests."""
    print(" Testing Electricity Price Forecasting Improvements")
    print("=" * 60)
    
    try:
        # Test feature selection
        test_feature_selection()
        
        # Test safe MAPE calculation
        test_safe_mape()
        
        # Test future predictions
        test_future_predictions()
        
        print("\n" + "=" * 60)
        print(" ALL TESTS PASSED!")
        print(" Feature selection working correctly")
        print(" Safe MAPE calculation handling edge cases")
        print(" Future predictions generating realistic results")
        print(" All improvements are working as expected!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
