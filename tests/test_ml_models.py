"""
Tests for machine learning models module.
"""

import pytest
import pandas as pd
import numpy as np
from src.models.ml_models import MLModels


class TestMLModels:
    """Test cases for MLModels class."""
    
    def test_init(self):
        """Test MLModels initialization."""
        ml_models = MLModels()
        assert ml_models is not None
        assert ml_models.random_state == 42
        assert len(ml_models.models) > 0
        assert ml_models.trained_models == {}
        assert ml_models.results == {}
    
    def test_initialize_models(self):
        """Test model initialization."""
        ml_models = MLModels()
        models = ml_models.models
        
        # Check that expected models are present
        expected_models = [
            'linear_regression', 'ridge', 'lasso', 'elastic_net',
            'random_forest', 'gradient_boosting', 'svr', 'mlp'
        ]
        
        for model_name in expected_models:
            assert model_name in models
            assert models[model_name] is not None
    
    def test_train_single_model(self, sample_train_test_data):
        """Test training a single model."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        ml_models = MLModels()
        
        # Train a simple model
        model_name = 'linear_regression'
        ml_models.train_model(model_name, X_train, y_train)
        
        assert model_name in ml_models.trained_models
        assert ml_models.trained_models[model_name] is not None
    
    def test_train_all_models(self, sample_train_test_data):
        """Test training all models."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        ml_models = MLModels()
        
        ml_models.train_all(X_train, y_train)
        
        # Check that all models are trained
        assert len(ml_models.trained_models) > 0
        for model_name in ml_models.models.keys():
            assert model_name in ml_models.trained_models
    
    def test_predict_single_model(self, sample_train_test_data):
        """Test prediction with a single model."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        ml_models = MLModels()
        
        # Train and predict
        model_name = 'linear_regression'
        ml_models.train_model(model_name, X_train, y_train)
        predictions = ml_models.predict_model(model_name, X_test)
        
        assert len(predictions) == len(X_test)
        assert not np.isnan(predictions).any()
        assert not np.isinf(predictions).any()
    
    def test_predict_all_models(self, sample_train_test_data):
        """Test prediction with all models."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        ml_models = MLModels()
        
        # Train and predict all
        ml_models.train_all(X_train, y_train)
        predictions = ml_models.predict_all(X_test)
        
        assert len(predictions) > 0
        for model_name, pred in predictions.items():
            assert len(pred) == len(X_test)
            assert not np.isnan(pred).any()
            assert not np.isinf(pred).any()
    
    def test_evaluate_single_model(self, sample_train_test_data):
        """Test evaluation of a single model."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        ml_models = MLModels()
        
        # Train, predict, and evaluate
        model_name = 'linear_regression'
        ml_models.train_model(model_name, X_train, y_train)
        predictions = ml_models.predict_model(model_name, X_test)
        results = ml_models.evaluate_model(y_test, predictions, model_name)
        
        assert 'rmse' in results
        assert 'mae' in results
        assert 'r2' in results
        assert results['rmse'] >= 0
        assert results['mae'] >= 0
    
    def test_evaluate_all_models(self, sample_train_test_data):
        """Test evaluation of all models."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        ml_models = MLModels()
        
        # Train, predict, and evaluate all
        ml_models.train_all(X_train, y_train)
        predictions = ml_models.predict_all(X_test)
        results = ml_models.evaluate_all(y_test, predictions)
        
        assert len(results) > 0
        for model_name, result in results.items():
            assert 'rmse' in result
            assert 'mae' in result
            assert 'r2' in result
    
    def test_preprocess_features(self, sample_train_test_data):
        """Test feature preprocessing."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        ml_models = MLModels()
        
        # Add some NaN values to test preprocessing
        X_train_with_nans = X_train.copy()
        X_train_with_nans.iloc[0, 0] = np.nan
        X_test_with_nans = X_test.copy()
        X_test_with_nans.iloc[0, 0] = np.nan
        
        X_train_processed, X_test_processed = ml_models.preprocess_features(
            X_train_with_nans, X_test_with_nans
        )
        
        # Check that NaN values are handled
        assert not X_train_processed.isnull().any().any()
        assert not X_test_processed.isnull().any().any()
        
        # Check that data is scaled
        assert X_train_processed.std().mean() < 2.0  # Should be scaled
    
    def test_hyperparameter_tuning(self, sample_train_test_data):
        """Test hyperparameter tuning."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        ml_models = MLModels()
        
        # Test with a simple model that has hyperparameters
        model_name = 'ridge'
        ml_models.train_model(model_name, X_train, y_train, tune_hyperparameters=True)
        
        assert model_name in ml_models.trained_models
        assert ml_models.trained_models[model_name] is not None
    
    def test_feature_importance(self, sample_train_test_data):
        """Test feature importance calculation."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        ml_models = MLModels()
        
        # Train a model that supports feature importance
        model_name = 'random_forest'
        ml_models.train_model(model_name, X_train, y_train)
        
        # Get feature importance
        importance = ml_models.get_feature_importance(model_name)
        
        if importance is not None:
            assert len(importance) == X_train.shape[1]
            assert all(imp >= 0 for imp in importance)
    
    def test_model_persistence(self, sample_train_test_data, tmp_path):
        """Test model saving and loading."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        ml_models = MLModels()
        
        # Train a model
        model_name = 'linear_regression'
        ml_models.train_model(model_name, X_train, y_train)
        
        # Save model
        model_path = tmp_path / f"{model_name}.pkl"
        ml_models.save_model(model_name, str(model_path))
        
        # Load model
        new_ml_models = MLModels()
        new_ml_models.load_model(model_name, str(model_path))
        
        # Test that loaded model works
        predictions = new_ml_models.predict_model(model_name, X_test)
        assert len(predictions) == len(X_test)
    
    def test_invalid_model_name(self, sample_train_test_data):
        """Test handling of invalid model names."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        ml_models = MLModels()
        
        with pytest.raises(ValueError):
            ml_models.train_model('invalid_model', X_train, y_train)
    
    def test_prediction_without_training(self, sample_train_test_data):
        """Test prediction without training."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        ml_models = MLModels()
        
        with pytest.raises(ValueError):
            ml_models.predict_model('linear_regression', X_test)
