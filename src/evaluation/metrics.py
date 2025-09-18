"""
Comprehensive evaluation metrics for electricity price forecasting.

This module provides various metrics specifically designed for evaluating
electricity price forecasting models, including both statistical and
business-oriented metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for electricity price forecasting.
    
    This class provides various metrics including statistical measures,
    directional accuracy, and business-oriented metrics for electricity
    price forecasting models.
    """
    
    def __init__(self):
        """Initialize evaluation metrics."""
        self.results = {}
    
    def calculate_all_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                            model_name: str = 'model') -> Dict[str, float]:
        """
        Calculate all available metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        
        # Basic statistical metrics
        metrics.update(self._calculate_statistical_metrics(y_true, y_pred))
        
        # Directional accuracy metrics
        metrics.update(self._calculate_directional_metrics(y_true, y_pred))
        
        # Business-oriented metrics
        metrics.update(self._calculate_business_metrics(y_true, y_pred))
        
        # Time series specific metrics
        metrics.update(self._calculate_timeseries_metrics(y_true, y_pred))
        
        self.results[model_name] = metrics
        return metrics
    
    def _calculate_statistical_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic statistical metrics."""
        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true.iloc[:min_len]
        y_pred = y_pred[:min_len]
        
        # Mean Absolute Error
        mae = mean_absolute_error(y_true, y_pred)
        
        # Root Mean Square Error
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Symmetric Mean Absolute Percentage Error
        smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
        
        # R-squared
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Scaled Error (MASE)
        # Using naive forecast as baseline
        naive_forecast = y_true.shift(1).dropna()
        naive_mae = mean_absolute_error(y_true.iloc[1:], naive_forecast)
        mase = mae / naive_mae if naive_mae > 0 else np.inf
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'smape': smape,
            'r2': r2,
            'mase': mase
        }
    
    def _calculate_directional_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate directional accuracy metrics."""
        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true.iloc[:min_len]
        y_pred = y_pred[:min_len]
        
        if len(y_true) < 2:
            return {'directional_accuracy': 0.0, 'directional_rmse': np.inf}
        
        # Directional accuracy
        y_diff = np.diff(y_true.values)
        pred_diff = np.diff(y_pred)
        
        # Correct direction predictions
        correct_direction = (y_diff * pred_diff) > 0
        directional_accuracy = np.mean(correct_direction) * 100
        
        # Directional RMSE (only for correct direction predictions)
        if np.any(correct_direction):
            directional_rmse = np.sqrt(mean_squared_error(
                y_true.iloc[1:][correct_direction],
                y_pred[1:][correct_direction]
            ))
        else:
            directional_rmse = np.inf
        
        return {
            'directional_accuracy': directional_accuracy,
            'directional_rmse': directional_rmse
        }
    
    def _calculate_business_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate business-oriented metrics."""
        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true.iloc[:min_len]
        y_pred = y_pred[:min_len]
        
        # Price level accuracy (within certain thresholds)
        price_errors = np.abs(y_true - y_pred)
        
        # Accuracy within 5% of actual price
        accuracy_5pct = np.mean(price_errors / y_true <= 0.05) * 100
        
        # Accuracy within 10% of actual price
        accuracy_10pct = np.mean(price_errors / y_true <= 0.10) * 100
        
        # Accuracy within 20% of actual price
        accuracy_20pct = np.mean(price_errors / y_true <= 0.20) * 100
        
        # Cost impact metrics (assuming 1 MWh consumption)
        cost_error = np.sum(np.abs(y_true - y_pred))  # Total cost error in €
        avg_cost_error = cost_error / len(y_true)  # Average cost error per hour
        
        # Peak price accuracy (for prices above 75th percentile)
        price_75th = np.percentile(y_true, 75)
        peak_mask = y_true >= price_75th
        if np.any(peak_mask):
            peak_mae = mean_absolute_error(y_true[peak_mask], y_pred[peak_mask])
            peak_mape = np.mean(np.abs((y_true[peak_mask] - y_pred[peak_mask]) / y_true[peak_mask])) * 100
        else:
            peak_mae = 0
            peak_mape = 0
        
        return {
            'accuracy_5pct': accuracy_5pct,
            'accuracy_10pct': accuracy_10pct,
            'accuracy_20pct': accuracy_20pct,
            'total_cost_error': cost_error,
            'avg_cost_error': avg_cost_error,
            'peak_mae': peak_mae,
            'peak_mape': peak_mape
        }
    
    def _calculate_timeseries_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate time series specific metrics."""
        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true.iloc[:min_len]
        y_pred = y_pred[:min_len]
        
        # Theil's U statistic
        naive_forecast = y_true.shift(1).dropna()
        if len(naive_forecast) > 0:
            naive_mse = mean_squared_error(y_true.iloc[1:], naive_forecast)
            model_mse = mean_squared_error(y_true, y_pred)
            theil_u = np.sqrt(model_mse / naive_mse) if naive_mse > 0 else np.inf
        else:
            theil_u = np.inf
        
        # Persistence model accuracy (same as yesterday)
        if len(y_true) > 24:
            persistence_forecast = y_true.shift(24).dropna()
            persistence_mae = mean_absolute_error(y_true.iloc[24:], persistence_forecast)
            model_mae = mean_absolute_error(y_true, y_pred)
            skill_score = 1 - (model_mae / persistence_mae) if persistence_mae > 0 else 0
        else:
            skill_score = 0
        
        # Volatility prediction accuracy
        y_true_vol = y_true.rolling(window=24).std()
        y_pred_vol = pd.Series(y_pred).rolling(window=24).std()
        vol_correlation = y_true_vol.corr(y_pred_vol) if len(y_true_vol.dropna()) > 0 else 0
        
        return {
            'theil_u': theil_u,
            'skill_score': skill_score,
            'volatility_correlation': vol_correlation
        }
    
    def compare_models(self, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compare multiple models based on their metrics.
        
        Args:
            results: Dictionary of model results
            
        Returns:
            DataFrame with model comparison
        """
        comparison_df = pd.DataFrame(results).T
        
        # Add ranking columns for key metrics
        key_metrics = ['rmse', 'mae', 'mape', 'directional_accuracy', 'r2']
        for metric in key_metrics:
            if metric in comparison_df.columns:
                if metric in ['rmse', 'mae', 'mape']:
                    # Lower is better
                    comparison_df[f'{metric}_rank'] = comparison_df[metric].rank(ascending=True)
                else:
                    # Higher is better
                    comparison_df[f'{metric}_rank'] = comparison_df[metric].rank(ascending=False)
        
        return comparison_df.sort_values('rmse')
    
    def get_model_ranking(self, metric: str = 'rmse') -> pd.DataFrame:
        """
        Get model ranking based on a specific metric.
        
        Args:
            metric: Metric to use for ranking
            
        Returns:
            DataFrame with model ranking
        """
        if not self.results:
            raise ValueError("No results available. Run calculate_all_metrics() first.")
        
        ranking_data = []
        for model_name, metrics in self.results.items():
            ranking_data.append({
                'model': model_name,
                'score': metrics.get(metric, np.inf)
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        
        if metric in ['rmse', 'mae', 'mape', 'mase', 'theil_u']:
            ranking_df = ranking_df.sort_values('score')
        else:
            ranking_df = ranking_df.sort_values('score', ascending=False)
        
        ranking_df['rank'] = range(1, len(ranking_df) + 1)
        return ranking_df
    
    def calculate_confidence_intervals(self, y_true: pd.Series, y_pred: np.ndarray, 
                                     confidence: float = 0.95) -> Dict[str, float]:
        """
        Calculate confidence intervals for predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            confidence: Confidence level (default: 0.95)
            
        Returns:
            Dictionary with confidence interval statistics
        """
        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true.iloc[:min_len]
        y_pred = y_pred[:min_len]
        
        errors = y_true - y_pred
        error_std = np.std(errors)
        
        # Z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        # Confidence interval width
        ci_width = z_score * error_std
        
        # Coverage (percentage of true values within confidence interval)
        lower_bound = y_pred - ci_width
        upper_bound = y_pred + ci_width
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound)) * 100
        
        return {
            'confidence_level': confidence,
            'ci_width': ci_width,
            'coverage': coverage,
            'error_std': error_std
        }
    
    def calculate_rolling_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                                 window: int = 24) -> pd.DataFrame:
        """
        Calculate rolling metrics over time.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            window: Rolling window size
            
        Returns:
            DataFrame with rolling metrics
        """
        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true.iloc[:min_len]
        y_pred = y_pred[:min_len]
        
        rolling_metrics = []
        
        for i in range(window, len(y_true)):
            y_window = y_true.iloc[i-window:i]
            pred_window = y_pred[i-window:i]
            
            mae = mean_absolute_error(y_window, pred_window)
            rmse = np.sqrt(mean_squared_error(y_window, pred_window))
            mape = np.mean(np.abs((y_window - pred_window) / y_window)) * 100
            
            rolling_metrics.append({
                'index': i,
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            })
        
        return pd.DataFrame(rolling_metrics)
    
    def export_results(self, filepath: str) -> None:
        """
        Export results to CSV file.
        
        Args:
            filepath: Path to save the results
        """
        if not self.results:
            raise ValueError("No results available to export.")
        
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv(filepath)
        logger.info(f"Results exported to {filepath}")


def main():
    """Example usage of evaluation metrics."""
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic electricity price data
    t = np.arange(n_samples)
    price = 50 + 10 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 2, n_samples)
    
    # Create true and predicted values
    y_true = pd.Series(price, name='price')
    y_pred = price + np.random.normal(0, 1, n_samples)  # Add some noise to predictions
    
    # Calculate metrics
    evaluator = EvaluationMetrics()
    metrics = evaluator.calculate_all_metrics(y_true, y_pred, 'test_model')
    
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Calculate confidence intervals
    ci_metrics = evaluator.calculate_confidence_intervals(y_true, y_pred)
    print(f"\nConfidence Intervals (95%):")
    print(f"Coverage: {ci_metrics['coverage']:.2f}%")
    print(f"CI Width: {ci_metrics['ci_width']:.2f}")


if __name__ == "__main__":
    main()
