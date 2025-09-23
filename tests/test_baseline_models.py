"""
Tests for baseline models module.
"""

import pytest
import pandas as pd
import numpy as np
from src.models.baseline_models import BaselineModels


class TestBaselineModels:
    """Test cases for BaselineModels class."""
    
    def test_init(self):
        """Test BaselineModels initialization."""
        baseline_models = BaselineModels()
        assert baseline_models is not None
        assert baseline_models.trained_models == {}
        assert baseline_models.results == {}
    
    def test_initialize_models(self):
        """Test baseline model initialization."""
        baseline_models = BaselineModels()
        models = baseline_models.models
        
        # Check that expected baseline models are present
        expected_models = [
            'naive', 'naive_24h', 'seasonal_naive', 'seasonal_naive_weekly',
            'mean', 'drift'
        ]
        
        for model_name in expected_models:
            assert model_name in models
            assert models[model_name] is not None
    
    def test_naive_forecaster(self, sample_train_test_data):
        """Test naive forecaster."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        baseline_models = BaselineModels()
        
        # Train naive model
        baseline_models.train_model('naive', X_train, y_train)
        
        # Make predictions
        predictions = baseline_models.predict_model('naive', X_test)
        
        assert len(predictions) == len(X_test)
        assert not np.isnan(predictions).any()
        # Naive should predict the last training value
        assert all(pred == y_train.iloc[-1] for pred in predictions)
    
    def test_seasonal_naive_forecaster(self, sample_train_test_data):
        """Test seasonal naive forecaster."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        baseline_models = BaselineModels()
        
        # Train seasonal naive model
        baseline_models.train_model('seasonal_naive', X_train, y_train)
        
        # Make predictions
        predictions = baseline_models.predict_model('seasonal_naive', X_test)
        
        assert len(predictions) == len(X_test)
        assert not np.isnan(predictions).any()
    
    def test_mean_forecaster(self, sample_train_test_data):
        """Test mean forecaster."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        baseline_models = BaselineModels()
        
        # Train mean model
        baseline_models.train_model('mean', X_train, y_train)
        
        # Make predictions
        predictions = baseline_models.predict_model('mean', X_test)
        
        assert len(predictions) == len(X_test)
        assert not np.isnan(predictions).any()
        # Mean should predict the training mean
        expected_mean = y_train.mean()
        assert all(abs(pred - expected_mean) < 1e-10 for pred in predictions)
    
    def test_drift_forecaster(self, sample_train_test_data):
        """Test drift forecaster."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        baseline_models = BaselineModels()
        
        # Train drift model
        baseline_models.train_model('drift', X_train, y_train)
        
        # Make predictions
        predictions = baseline_models.predict_model('drift', X_test)
        
        assert len(predictions) == len(X_test)
        assert not np.isnan(predictions).any()
    
    def test_train_all_models(self, sample_train_test_data):
        """Test training all baseline models."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        baseline_models = BaselineModels()
        
        baseline_models.train_all(X_train, y_train)
        
        # Check that all models are trained
        assert len(baseline_models.trained_models) > 0
        for model_name in baseline_models.models.keys():
            assert model_name in baseline_models.trained_models
    
    def test_predict_all_models(self, sample_train_test_data):
        """Test prediction with all baseline models."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        baseline_models = BaselineModels()
        
        # Train and predict all
        baseline_models.train_all(X_train, y_train)
        predictions = baseline_models.predict_all(X_test)
        
        assert len(predictions) > 0
        for model_name, pred in predictions.items():
            assert len(pred) == len(X_test)
            assert not np.isnan(pred).any()
            assert not np.isinf(pred).any()
    
    def test_evaluate_single_model(self, sample_train_test_data):
        """Test evaluation of a single baseline model."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        baseline_models = BaselineModels()
        
        # Train, predict, and evaluate
        model_name = 'naive'
        baseline_models.train_model(model_name, X_train, y_train)
        predictions = baseline_models.predict_model(model_name, X_test)
        results = baseline_models.evaluate_model(y_test, predictions, model_name)
        
        assert 'rmse' in results
        assert 'mae' in results
        assert 'r2' in results
        assert results['rmse'] >= 0
        assert results['mae'] >= 0
    
    def test_evaluate_all_models(self, sample_train_test_data):
        """Test evaluation of all baseline models."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        baseline_models = BaselineModels()
        
        # Train, predict, and evaluate all
        baseline_models.train_all(X_train, y_train)
        predictions = baseline_models.predict_all(X_test)
        results = baseline_models.evaluate_all(y_test, predictions)
        
        assert len(results) > 0
        for model_name, result in results.items():
            assert 'rmse' in result
            assert 'mae' in result
            assert 'r2' in result
    
    def test_empty_training_data(self):
        """Test handling of empty training data."""
        baseline_models = BaselineModels()
        empty_X = pd.DataFrame()
        empty_y = pd.Series(dtype=float)
        
        with pytest.raises(ValueError):
            baseline_models.train_model('naive', empty_X, empty_y)
    
    def test_single_sample_training_data(self):
        """Test handling of single sample training data."""
        baseline_models = BaselineModels()
        single_X = pd.DataFrame({'feature': [1]})
        single_y = pd.Series([10])
        
        # Some models should work with single sample
        baseline_models.train_model('naive', single_X, single_y)
        assert 'naive' in baseline_models.trained_models
    
    def test_invalid_model_name(self, sample_train_test_data):
        """Test handling of invalid model names."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        baseline_models = BaselineModels()
        
        with pytest.raises(ValueError):
            baseline_models.train_model('invalid_model', X_train, y_train)
    
    def test_prediction_without_training(self, sample_train_test_data):
        """Test prediction without training."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        baseline_models = BaselineModels()
        
        with pytest.raises(ValueError):
            baseline_models.predict_model('naive', X_test)
    
    def test_seasonal_naive_with_insufficient_data(self):
        """Test seasonal naive with insufficient data for seasonality."""
        baseline_models = BaselineModels()
        
        # Create data with less than 24 hours
        short_X = pd.DataFrame({'feature': range(10)})
        short_y = pd.Series(range(10))
        
        # Should still work but may not be meaningful
        baseline_models.train_model('seasonal_naive', short_X, short_y)
        assert 'seasonal_naive' in baseline_models.trained_models
