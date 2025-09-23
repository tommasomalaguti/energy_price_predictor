"""
Tests for evaluation metrics module.
"""

import pytest
import pandas as pd
import numpy as np
from src.evaluation.metrics import EvaluationMetrics


class TestEvaluationMetrics:
    """Test cases for EvaluationMetrics class."""
    
    def test_init(self):
        """Test EvaluationMetrics initialization."""
        metrics = EvaluationMetrics()
        assert metrics is not None
    
    def test_rmse_calculation(self):
        """Test RMSE calculation."""
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = EvaluationMetrics()
        results = metrics.calculate_all_metrics(y_true, y_pred)
        
        expected_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        assert abs(results['rmse'] - expected_rmse) < 1e-10
        assert results['rmse'] >= 0
    
    def test_mae_calculation(self):
        """Test MAE calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        mae = EvaluationMetrics.calculate_mae(y_true, y_pred)
        expected_mae = np.mean(np.abs(y_true - y_pred))
        
        assert abs(mae - expected_mae) < 1e-10
        assert mae >= 0
    
    def test_mape_calculation(self):
        """Test MAPE calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        mape = EvaluationMetrics.calculate_mape(y_true, y_pred)
        expected_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        assert abs(mape - expected_mape) < 1e-10
        assert mape >= 0
    
    def test_mape_with_zeros(self):
        """Test MAPE calculation with zero values in y_true."""
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0.1, 1.1, 1.9, 3.1, 3.9])
        
        mape = EvaluationMetrics.calculate_mape(y_true, y_pred)
        # Should handle zeros gracefully
        assert not np.isnan(mape)
        assert not np.isinf(mape)
    
    def test_r2_calculation(self):
        """Test R² calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        r2 = EvaluationMetrics.calculate_r2(y_true, y_pred)
        expected_r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        
        assert abs(r2 - expected_r2) < 1e-10
        assert r2 <= 1
    
    def test_directional_accuracy(self):
        """Test directional accuracy calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        # All predictions have correct direction
        dir_acc = EvaluationMetrics.calculate_directional_accuracy(y_true, y_pred)
        assert dir_acc == 100.0
        
        # Test with wrong directions
        y_pred_wrong = np.array([0.9, 2.1, 2.9, 4.1, 4.9])
        dir_acc_wrong = EvaluationMetrics.calculate_directional_accuracy(y_true, y_pred_wrong)
        assert dir_acc_wrong == 0.0
    
    def test_directional_accuracy_with_constant(self):
        """Test directional accuracy with constant values."""
        y_true = np.array([1, 1, 1, 1, 1])
        y_pred = np.array([1.1, 1.1, 1.1, 1.1, 1.1])
        
        dir_acc = EvaluationMetrics.calculate_directional_accuracy(y_true, y_pred)
        # Should handle constant values gracefully
        assert not np.isnan(dir_acc)
    
    def test_mase_calculation(self):
        """Test MASE calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        
        mase = EvaluationMetrics.calculate_mase(y_true, y_pred, y_train)
        assert mase >= 0
        assert not np.isnan(mase)
        assert not np.isinf(mase)
    
    def test_mase_with_constant_naive(self):
        """Test MASE calculation with constant naive predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        y_train = np.array([1, 1, 1, 1, 1])  # Constant training data
        
        mase = EvaluationMetrics.calculate_mase(y_true, y_pred, y_train)
        # Should handle constant naive predictions gracefully
        assert not np.isnan(mase)
        assert not np.isinf(mase)
    
    def test_evaluate_single_model(self):
        """Test evaluation of a single model."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        
        results = EvaluationMetrics.evaluate_single_model(y_true, y_pred, y_train, "test_model")
        
        assert isinstance(results, dict)
        assert 'rmse' in results
        assert 'mae' in results
        assert 'mape' in results
        assert 'r2' in results
        assert 'directional_accuracy' in results
        assert 'mase' in results
        
        # Check that all values are valid
        for metric, value in results.items():
            assert not np.isnan(value)
            assert not np.isinf(value)
    
    def test_evaluate_multiple_models(self):
        """Test evaluation of multiple models."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        
        predictions = {
            'model1': np.array([1.1, 1.9, 3.1, 3.9, 5.1]),
            'model2': np.array([0.9, 2.1, 2.9, 4.1, 4.9]),
            'model3': np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        }
        
        results = EvaluationMetrics.evaluate_multiple_models(y_true, predictions, y_train)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3
        assert 'rmse' in results.columns
        assert 'mae' in results.columns
        assert 'mape' in results.columns
        assert 'r2' in results.columns
        assert 'directional_accuracy' in results.columns
        assert 'mase' in results.columns
        
        # Check that all values are valid
        for col in results.columns:
            assert not results[col].isna().any()
            assert not np.isinf(results[col]).any()
    
    def test_empty_predictions(self):
        """Test evaluation with empty predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        
        with pytest.raises(ValueError):
            EvaluationMetrics.evaluate_single_model(y_true, np.array([]), y_train, "test_model")
    
    def test_mismatched_lengths(self):
        """Test evaluation with mismatched array lengths."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1])  # Different length
        y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        
        with pytest.raises(ValueError):
            EvaluationMetrics.evaluate_single_model(y_true, y_pred, y_train, "test_model")
    
    def test_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])  # Perfect predictions
        y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        
        results = EvaluationMetrics.evaluate_single_model(y_true, y_pred, y_train, "perfect_model")
        
        assert results['rmse'] == 0.0
        assert results['mae'] == 0.0
        assert results['mape'] == 0.0
        assert results['r2'] == 1.0
        assert results['directional_accuracy'] == 100.0
    
    def test_very_bad_predictions(self):
        """Test evaluation with very bad predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([10, 20, 30, 40, 50])  # Very bad predictions
        y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        
        results = EvaluationMetrics.evaluate_single_model(y_true, y_pred, y_train, "bad_model")
        
        assert results['rmse'] > 0
        assert results['mae'] > 0
        assert results['mape'] > 0
        assert results['r2'] < 0  # Negative R² for very bad predictions
        assert results['directional_accuracy'] == 100.0  # Still correct direction
