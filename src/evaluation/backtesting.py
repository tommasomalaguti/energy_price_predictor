"""
Backtesting framework for electricity price forecasting models.

This module provides comprehensive backtesting capabilities including
walk-forward analysis, performance tracking, and risk assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    """
    Comprehensive backtesting framework for electricity price forecasting.
    """
    
    def __init__(self, initial_capital: float = 10000, transaction_cost: float = 0.001,
                 risk_free_rate: float = 0.02):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Initial capital for backtesting
            transaction_cost: Transaction cost as fraction of trade value
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        self.results = {}
        
    def walk_forward_analysis(self, model, X: pd.DataFrame, y: pd.Series,
                            train_window: int = 1000, test_window: int = 24,
                            step_size: int = 24) -> Dict[str, Any]:
        """
        Perform walk-forward analysis for model evaluation.
        
        Args:
            model: Forecasting model
            X: Feature matrix
            y: Target series
            train_window: Size of training window
            test_window: Size of test window
            step_size: Step size for moving window
            
        Returns:
            Dictionary with backtesting results
        """
        logger.info("Starting walk-forward analysis...")
        
        predictions = []
        actuals = []
        timestamps = []
        model_performance = []
        
        # Walk-forward analysis
        for start_idx in range(train_window, len(X) - test_window + 1, step_size):
            end_idx = start_idx + test_window
            
            # Training data
            X_train = X.iloc[start_idx - train_window:start_idx]
            y_train = y.iloc[start_idx - train_window:start_idx]
            
            # Test data
            X_test = X.iloc[start_idx:end_idx]
            y_test = y.iloc[start_idx:end_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Store results
            predictions.extend(y_pred)
            actuals.extend(y_test.values)
            timestamps.extend(X_test.index)
            
            # Calculate performance for this window
            mse = np.mean((y_test - y_pred) ** 2)
            mae = np.mean(np.abs(y_test - y_pred))
            model_performance.append({
                'window_start': start_idx,
                'window_end': end_idx,
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse)
            })
        
        # Store results
        self.results['walk_forward'] = {
            'predictions': np.array(predictions),
            'actuals': np.array(actuals),
            'timestamps': timestamps,
            'model_performance': model_performance
        }
        
        logger.info(f"Walk-forward analysis completed: {len(predictions)} predictions")
        return self.results['walk_forward']
    
    def rolling_window_analysis(self, model, X: pd.DataFrame, y: pd.Series,
                              window_size: int = 1000, step_size: int = 24) -> Dict[str, Any]:
        """
        Perform rolling window analysis.
        
        Args:
            model: Forecasting model
            X: Feature matrix
            y: Target series
            window_size: Size of rolling window
            step_size: Step size for rolling window
            
        Returns:
            Dictionary with rolling window results
        """
        logger.info("Starting rolling window analysis...")
        
        predictions = []
        actuals = []
        timestamps = []
        performance_over_time = []
        
        # Rolling window analysis
        for start_idx in range(window_size, len(X), step_size):
            end_idx = min(start_idx + step_size, len(X))
            
            # Training data (rolling window)
            X_train = X.iloc[start_idx - window_size:start_idx]
            y_train = y.iloc[start_idx - window_size:start_idx]
            
            # Test data
            X_test = X.iloc[start_idx:end_idx]
            y_test = y.iloc[start_idx:end_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Store results
            predictions.extend(y_pred)
            actuals.extend(y_test.values)
            timestamps.extend(X_test.index)
            
            # Calculate performance
            mse = np.mean((y_test - y_pred) ** 2)
            mae = np.mean(np.abs(y_test - y_pred))
            performance_over_time.append({
                'timestamp': X_test.index[0],
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse)
            })
        
        # Store results
        self.results['rolling_window'] = {
            'predictions': np.array(predictions),
            'actuals': np.array(actuals),
            'timestamps': timestamps,
            'performance_over_time': performance_over_time
        }
        
        logger.info(f"Rolling window analysis completed: {len(predictions)} predictions")
        return self.results['rolling_window']
    
    def calculate_performance_metrics(self, predictions: np.ndarray, 
                                     actuals: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            predictions: Model predictions
            actuals: Actual values
            
        Returns:
            Dictionary with performance metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Basic metrics
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        r2 = r2_score(actuals, predictions)
        
        # Directional accuracy
        if len(predictions) > 1:
            actual_diff = np.diff(actuals)
            pred_diff = np.diff(predictions)
            directional_accuracy = np.mean((actual_diff * pred_diff) > 0) * 100
        else:
            directional_accuracy = 0
        
        # Theil's U statistic
        naive_forecast = np.roll(actuals, 1)
        naive_forecast[0] = actuals[0]
        naive_mse = mean_squared_error(actuals[1:], naive_forecast[1:])
        theil_u = np.sqrt(mse / naive_mse) if naive_mse > 0 else np.inf
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'theil_u': theil_u
        }
    
    def calculate_trading_metrics(self, predictions: np.ndarray, 
                                 actuals: np.ndarray,
                                 trading_strategy: str = 'simple') -> Dict[str, float]:
        """
        Calculate trading-specific metrics.
        
        Args:
            predictions: Model predictions
            actuals: Actual values
            trading_strategy: Trading strategy
            
        Returns:
            Dictionary with trading metrics
        """
        if trading_strategy == 'simple':
            return self._calculate_simple_trading_metrics(predictions, actuals)
        elif trading_strategy == 'arbitrage':
            return self._calculate_arbitrage_metrics(predictions, actuals)
        else:
            raise ValueError(f"Unknown trading strategy: {trading_strategy}")
    
    def _calculate_simple_trading_metrics(self, predictions: np.ndarray, 
                                         actuals: np.ndarray) -> Dict[str, float]:
        """Calculate simple trading metrics."""
        # Simulate trading based on predictions
        positions = np.zeros(len(predictions))
        returns = np.zeros(len(predictions))
        
        # Simple strategy: buy when predicted price is low, sell when high
        price_median = np.median(actuals)
        
        for i in range(1, len(predictions)):
            if predictions[i] < price_median * 0.9:  # Buy signal
                positions[i] = 1
            elif predictions[i] > price_median * 1.1:  # Sell signal
                positions[i] = -1
            else:
                positions[i] = positions[i-1]  # Hold
            
            # Calculate returns
            if positions[i-1] != 0:
                returns[i] = positions[i-1] * (actuals[i] - actuals[i-1]) / actuals[i-1]
        
        # Trading metrics
        total_return = np.sum(returns)
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)  # Annualized
        max_drawdown = self._calculate_max_drawdown(np.cumsum(returns))
        win_rate = np.mean(returns > 0) * 100
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }
    
    def _calculate_arbitrage_metrics(self, predictions: np.ndarray, 
                                    actuals: np.ndarray) -> Dict[str, float]:
        """Calculate arbitrage trading metrics."""
        # Find arbitrage opportunities
        price_spreads = actuals.max() - actuals.min()
        predicted_spreads = predictions.max() - predictions.min()
        
        # Arbitrage accuracy
        spread_accuracy = 1 - abs(price_spreads - predicted_spreads) / price_spreads if price_spreads > 0 else 0
        
        # Peak and valley detection
        actual_peaks = self._find_peaks(actuals)
        predicted_peaks = self._find_peaks(predictions)
        peak_detection_accuracy = len(set(actual_peaks) & set(predicted_peaks)) / max(len(actual_peaks), 1)
        
        return {
            'spread_accuracy': spread_accuracy,
            'peak_detection_accuracy': peak_detection_accuracy
        }
    
    def _find_peaks(self, series: np.ndarray, window: int = 3) -> List[int]:
        """Find peaks in time series."""
        peaks = []
        for i in range(window, len(series) - window):
            if (series[i] > series[i-window:i]).all() and (series[i] > series[i+1:i+window+1]).all():
                peaks.append(i)
        return peaks
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        return np.min(drawdown)
    
    def plot_backtest_results(self, analysis_type: str = 'walk_forward') -> None:
        """
        Plot backtesting results.
        
        Args:
            analysis_type: Type of analysis to plot
        """
        if analysis_type not in self.results:
            logger.warning(f"No results available for {analysis_type}")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            results = self.results[analysis_type]
            predictions = results['predictions']
            actuals = results['actuals']
            timestamps = results['timestamps']
            
            # Create time index
            if isinstance(timestamps[0], pd.Timestamp):
                time_index = timestamps
            else:
                time_index = range(len(predictions))
            
            # Plot predictions vs actuals
            plt.figure(figsize=(15, 10))
            
            # Main plot
            plt.subplot(2, 1, 1)
            plt.plot(time_index, actuals, 'b-', label='Actual', alpha=0.7, linewidth=1)
            plt.plot(time_index, predictions, 'r-', label='Predicted', alpha=0.7, linewidth=1)
            plt.title(f'Backtesting Results - {analysis_type.title()}')
            plt.xlabel('Time')
            plt.ylabel('Price (€/MWh)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Residuals plot
            plt.subplot(2, 1, 2)
            residuals = actuals - predictions
            plt.plot(time_index, residuals, 'g-', alpha=0.7, linewidth=1)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            plt.title('Prediction Residuals')
            plt.xlabel('Time')
            plt.ylabel('Residual (€/MWh)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
    
    def generate_report(self, analysis_type: str = 'walk_forward') -> Dict[str, Any]:
        """
        Generate comprehensive backtesting report.
        
        Args:
            analysis_type: Type of analysis to report
            
        Returns:
            Dictionary with comprehensive report
        """
        if analysis_type not in self.results:
            raise ValueError(f"No results available for {analysis_type}")
        
        results = self.results[analysis_type]
        predictions = results['predictions']
        actuals = results['actuals']
        
        # Calculate metrics
        performance_metrics = self.calculate_performance_metrics(predictions, actuals)
        trading_metrics = self.calculate_trading_metrics(predictions, actuals)
        
        # Generate report
        report = {
            'analysis_type': analysis_type,
            'n_predictions': len(predictions),
            'performance_metrics': performance_metrics,
            'trading_metrics': trading_metrics,
            'summary': {
                'best_metric': max(performance_metrics.items(), key=lambda x: x[1] if x[0] != 'theil_u' else -x[1]),
                'worst_metric': min(performance_metrics.items(), key=lambda x: x[1] if x[0] != 'theil_u' else -x[1])
            }
        }
        
        return report


def main():
    """Example usage of backtesting with real data."""
    print("Backtesting example requires real electricity price data.")
    print("Please use the ENTSO-E downloader to get real data first.")
    print("Example usage:")
    print("1. Download real data using ENTSOEDownloader")
    print("2. Preprocess data using DataPreprocessor") 
    print("3. Create Backtester instance")
    print("4. Run walk_forward_analysis() or rolling_window_analysis()")
    print("5. Generate report using generate_report()")


if __name__ == "__main__":
    main()
