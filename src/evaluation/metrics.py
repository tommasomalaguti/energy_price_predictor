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
        # Validate inputs
        if len(y_true) == 0:
            raise ValueError("y_true cannot be empty")
        if len(y_pred) == 0:
            raise ValueError("y_pred cannot be empty")
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true has {len(y_true)} elements, y_pred has {len(y_pred)} elements")
        
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
        
        # Mean Absolute Percentage Error (handle zeros by using symmetric MAPE)
        # For zero values, use symmetric MAPE to avoid division by zero
        mask = y_true != 0
        if np.any(mask):
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            # If all values are zero, use symmetric MAPE
            mape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
        
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
        
        # Correct direction predictions (both must have same sign)
        correct_direction = (y_diff * pred_diff) > 0
        # Handle case where all differences are zero (constant values)
        if len(correct_direction) == 0:
            directional_accuracy = 0.0
        else:
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
    
    def calculate_trading_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                                 trading_strategy: str = 'simple') -> Dict[str, float]:
        """
        Calculate trading-specific metrics for electricity price forecasting.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            trading_strategy: Trading strategy ('simple', 'arbitrage', 'peak_shaving')
            
        Returns:
            Dictionary with trading metrics
        """
        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true.iloc[:min_len]
        y_pred = y_pred[:min_len]
        
        if trading_strategy == 'simple':
            return self._calculate_simple_trading_metrics(y_true, y_pred)
        elif trading_strategy == 'arbitrage':
            return self._calculate_arbitrage_metrics(y_true, y_pred)
        elif trading_strategy == 'peak_shaving':
            return self._calculate_peak_shaving_metrics(y_true, y_pred)
        else:
            raise ValueError(f"Unknown trading strategy: {trading_strategy}")
    
    def _calculate_simple_trading_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate simple trading metrics."""
        # Price level accuracy for trading decisions
        price_errors = np.abs(y_true - y_pred)
        
        # Trading accuracy (within 5% for profitable trading)
        trading_accuracy_5pct = np.mean(price_errors / y_true <= 0.05) * 100
        
        # Peak price prediction accuracy (critical for trading)
        price_75th = np.percentile(y_true, 75)
        peak_mask = y_true >= price_75th
        peak_accuracy = np.mean(price_errors[peak_mask] / y_true[peak_mask] <= 0.10) * 100 if np.any(peak_mask) else 0
        
        # Volatility prediction accuracy
        y_true_vol = y_true.rolling(window=24).std()
        y_pred_vol = pd.Series(y_pred).rolling(window=24).std()
        vol_correlation = y_true_vol.corr(y_pred_vol) if len(y_true_vol.dropna()) > 0 else 0
        
        return {
            'trading_accuracy_5pct': trading_accuracy_5pct,
            'peak_prediction_accuracy': peak_accuracy,
            'volatility_correlation': vol_correlation
        }
    
    def _calculate_arbitrage_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate arbitrage trading metrics."""
        # Price spread prediction
        price_spread_true = y_true.max() - y_true.min()
        price_spread_pred = np.max(y_pred) - np.min(y_pred)
        spread_accuracy = 1 - abs(price_spread_true - price_spread_pred) / price_spread_true if price_spread_true > 0 else 0
        
        # Peak and valley prediction accuracy
        y_true_peaks = y_true.rolling(window=3, center=True).max() == y_true
        y_pred_peaks = pd.Series(y_pred).rolling(window=3, center=True).max() == pd.Series(y_pred)
        peak_detection_accuracy = np.mean(y_true_peaks == y_pred_peaks) * 100
        
        return {
            'spread_accuracy': spread_accuracy,
            'peak_detection_accuracy': peak_detection_accuracy
        }
    
    def _calculate_peak_shaving_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate peak shaving metrics."""
        # Peak hours identification
        price_75th = np.percentile(y_true, 75)
        peak_hours_true = y_true >= price_75th
        peak_hours_pred = y_pred >= price_75th
        
        peak_identification_accuracy = np.mean(peak_hours_true == peak_hours_pred) * 100
        
        # Peak magnitude prediction
        peak_prices_true = y_true[peak_hours_true]
        peak_prices_pred = y_pred[peak_hours_true]
        peak_magnitude_mape = np.mean(np.abs((peak_prices_true - peak_prices_pred) / peak_prices_true)) * 100 if len(peak_prices_true) > 0 else 0
        
        return {
            'peak_identification_accuracy': peak_identification_accuracy,
            'peak_magnitude_mape': peak_magnitude_mape
        }
    
    def calculate_risk_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate risk management metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with risk metrics
        """
        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true.iloc[:min_len]
        y_pred = y_pred[:min_len]
        
        # Value at Risk (VaR) - 95% confidence
        errors = y_true - y_pred
        var_95 = np.percentile(errors, 5)  # 5th percentile of errors
        
        # Expected Shortfall (Conditional VaR)
        es_95 = np.mean(errors[errors <= var_95])
        
        # Maximum drawdown in predictions
        cumulative_errors = np.cumsum(errors)
        running_max = np.maximum.accumulate(cumulative_errors)
        drawdown = cumulative_errors - running_max
        max_drawdown = np.min(drawdown)
        
        # Prediction stability (low variance in errors)
        error_stability = 1 / (np.var(errors) + 1e-8)
        
        return {
            'var_95': var_95,
            'expected_shortfall_95': es_95,
            'max_drawdown': max_drawdown,
            'error_stability': error_stability
        }
    
    def calculate_operational_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                                    operational_thresholds: Dict[str, float] = None) -> Dict[str, float]:
        """
        Calculate operational metrics for energy management.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            operational_thresholds: Thresholds for operational decisions
            
        Returns:
            Dictionary with operational metrics
        """
        if operational_thresholds is None:
            operational_thresholds = {
                'high_price_threshold': 80.0,  # €/MWh
                'low_price_threshold': 20.0,  # €/MWh
                'critical_threshold': 100.0   # €/MWh
            }
        
        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true.iloc[:min_len]
        y_pred = y_pred[:min_len]
        
        # High price event prediction
        high_price_true = y_true >= operational_thresholds['high_price_threshold']
        high_price_pred = y_pred >= operational_thresholds['high_price_threshold']
        high_price_accuracy = np.mean(high_price_true == high_price_pred) * 100
        
        # Critical price event prediction
        critical_price_true = y_true >= operational_thresholds['critical_threshold']
        critical_price_pred = y_pred >= operational_thresholds['critical_threshold']
        critical_price_accuracy = np.mean(critical_price_true == critical_price_pred) * 100
        
        # Low price event prediction
        low_price_true = y_true <= operational_thresholds['low_price_threshold']
        low_price_pred = y_pred <= operational_thresholds['low_price_threshold']
        low_price_accuracy = np.mean(low_price_true == low_price_pred) * 100
        
        # Operational cost impact
        cost_impact = np.sum(np.abs(y_true - y_pred))  # Total cost error
        
        return {
            'high_price_accuracy': high_price_accuracy,
            'critical_price_accuracy': critical_price_accuracy,
            'low_price_accuracy': low_price_accuracy,
            'total_cost_impact': cost_impact
        }


def main():
    """Example usage of evaluation metrics with real data."""
    print("Evaluation Metrics example requires real electricity price data.")
    print("Please use real data from ENTSO-E API or other sources.")
    print("Example usage:")
    print("1. Get real price data (y_true) and model predictions (y_pred)")
    print("2. Create EvaluationMetrics instance")
    print("3. Calculate metrics using calculate_all_metrics(y_true, y_pred)")
    print("4. Access individual metrics from the results dictionary")
    print("5. Use compare_models() to compare multiple models")


if __name__ == "__main__":
    main()
