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
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = EvaluationMetrics()
        results = metrics.calculate_all_metrics(y_true, y_pred)
        mae = results['mae']
        expected_mae = np.mean(np.abs(y_true - y_pred))
        
        assert abs(mae - expected_mae) < 1e-10
        assert mae >= 0
    
    def test_mape_calculation(self):
        """Test MAPE calculation."""
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = EvaluationMetrics()
        results = metrics.calculate_all_metrics(y_true, y_pred)
        mape = results['mape']
        expected_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        assert abs(mape - expected_mape) < 1e-10
        assert mape >= 0
    
    def test_mape_with_zeros(self):
        """Test MAPE calculation with zero values in y_true."""
        y_true = pd.Series([0, 1, 2, 3, 4])
        y_pred = np.array([0.1, 1.1, 1.9, 3.1, 3.9])
        
        metrics = EvaluationMetrics()
        results = metrics.calculate_all_metrics(y_true, y_pred)
        mape = results['mape']
        # Should handle zeros gracefully
        assert not np.isnan(mape)
        assert not np.isinf(mape)
    
    def test_r2_calculation(self):
        """Test R² calculation."""
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = EvaluationMetrics()
        results = metrics.calculate_all_metrics(y_true, y_pred)
        r2 = results['r2']
        expected_r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        
        assert abs(r2 - expected_r2) < 1e-10
        assert r2 <= 1
    
    def test_directional_accuracy(self):
        """Test directional accuracy calculation."""
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = EvaluationMetrics()
        results = metrics.calculate_all_metrics(y_true, y_pred)
        dir_acc = results['directional_accuracy']
        assert dir_acc == 100.0
        
        # Test with wrong directions (decreasing when should be increasing)
        y_pred_wrong = np.array([2.0, 1.5, 1.0, 0.5, 0.0])
        results_wrong = metrics.calculate_all_metrics(y_true, y_pred_wrong)
        dir_acc_wrong = results_wrong['directional_accuracy']
        assert dir_acc_wrong == 0.0
    
    def test_directional_accuracy_with_constant(self):
        """Test directional accuracy with constant values."""
        y_true = pd.Series([1, 1, 1, 1, 1])
        y_pred = np.array([1.1, 1.1, 1.1, 1.1, 1.1])
        
        metrics = EvaluationMetrics()
        results = metrics.calculate_all_metrics(y_true, y_pred)
        dir_acc = results['directional_accuracy']
        # Should handle constant values gracefully
        assert not np.isnan(dir_acc)
    
    def test_mase_calculation(self):
        """Test MASE calculation."""
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        y_train = pd.Series([0.5, 1.5, 2.5, 3.5, 4.5])
        
        metrics = EvaluationMetrics()
        results = metrics.calculate_all_metrics(y_true, y_pred)
        mase = results['mase']
        assert mase >= 0
        assert not np.isnan(mase)
        assert not np.isinf(mase)
    
    def test_mase_with_constant_naive(self):
        """Test MASE calculation with constant naive predictions."""
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        y_train = pd.Series([1, 1, 1, 1, 1])  # Constant training data
        
        metrics = EvaluationMetrics()
        results = metrics.calculate_all_metrics(y_true, y_pred)
        mase = results['mase']
        # Should handle constant naive predictions gracefully
        assert not np.isnan(mase)
        assert not np.isinf(mase)
    
    def test_evaluate_single_model(self):
        """Test evaluation of a single model."""
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = EvaluationMetrics()
        results = metrics.calculate_all_metrics(y_true, y_pred, "test_model")
        
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
        y_true = pd.Series([1, 2, 3, 4, 5])
        
        predictions = {
            'model1': np.array([1.1, 1.9, 3.1, 3.9, 5.1]),
            'model2': np.array([0.9, 2.1, 2.9, 4.1, 4.9]),
            'model3': np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        }
        
        metrics = EvaluationMetrics()
        results_list = []
        for model_name, y_pred in predictions.items():
            results = metrics.calculate_all_metrics(y_true, y_pred, model_name)
            results_list.append(results)
        
        # Convert to DataFrame for comparison
        results_df = pd.DataFrame(results_list)
        
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 3
        assert 'rmse' in results_df.columns
        assert 'mae' in results_df.columns
        assert 'mape' in results_df.columns
        assert 'r2' in results_df.columns
        assert 'directional_accuracy' in results_df.columns
        assert 'mase' in results_df.columns
        
        # Check that all values are valid
        for col in results_df.columns:
            assert not results_df[col].isna().any()
            assert not np.isinf(results_df[col]).any()
    
    def test_empty_predictions(self):
        """Test evaluation with empty predictions."""
        y_true = pd.Series([1, 2, 3, 4, 5])
        
        with pytest.raises(ValueError):
            metrics = EvaluationMetrics()
            metrics.calculate_all_metrics(y_true, np.array([]), "test_model")
    
    def test_mismatched_lengths(self):
        """Test evaluation with mismatched array lengths."""
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1])  # Different length
        
        with pytest.raises(ValueError):
            metrics = EvaluationMetrics()
            metrics.calculate_all_metrics(y_true, y_pred, "test_model")
    
    def test_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])  # Perfect predictions
        
        metrics = EvaluationMetrics()
        results = metrics.calculate_all_metrics(y_true, y_pred, "perfect_model")
        
        assert results['rmse'] == 0.0
        assert results['mae'] == 0.0
        assert results['mape'] == 0.0
        assert results['r2'] == 1.0
        assert results['directional_accuracy'] == 100.0
    
    def test_very_bad_predictions(self):
        """Test evaluation with very bad predictions."""
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = np.array([10, 20, 30, 40, 50])  # Very bad predictions
        
        metrics = EvaluationMetrics()
        results = metrics.calculate_all_metrics(y_true, y_pred, "bad_model")
        
        assert results['rmse'] > 0
        assert results['mae'] > 0
        assert results['mape'] > 0
        assert results['r2'] < 0  # Negative R² for very bad predictions
        assert results['directional_accuracy'] == 100.0  # Still correct direction
