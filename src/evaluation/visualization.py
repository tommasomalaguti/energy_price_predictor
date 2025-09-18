"""
Visualization tools for electricity price forecasting model evaluation.

This module provides comprehensive visualization capabilities for
analyzing model performance, comparing different models, and
understanding forecasting patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ModelVisualization:
    """
    Comprehensive visualization tools for electricity price forecasting models.
    
    This class provides various plotting functions for model evaluation,
    comparison, and analysis of electricity price forecasting results.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualization tools.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    def plot_predictions_vs_actual(self, y_true: pd.Series, predictions: Dict[str, np.ndarray],
                                 title: str = "Predictions vs Actual", 
                                 max_points: int = 1000,
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> None:
        """
        Plot predictions vs actual values for multiple models.
        
        Args:
            y_true: True values
            predictions: Dictionary of model predictions
            title: Plot title
            max_points: Maximum number of points to plot
            start_date: Start date for plotting
            end_date: End date for plotting
        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Limit points for plotting
        if max_points and len(y_true) > max_points:
            step = len(y_true) // max_points
            indices = np.arange(0, len(y_true), step)
        else:
            indices = np.arange(len(y_true))
        
        # Plot 1: Time series comparison
        axes[0].plot(indices, y_true.iloc[indices], label='Actual', linewidth=2, alpha=0.8)
        
        for i, (model_name, pred) in enumerate(predictions.items()):
            if len(pred) == len(y_true):
                axes[0].plot(indices, pred[indices], label=model_name, 
                           alpha=0.7, color=self.colors[i % len(self.colors)])
        
        axes[0].set_title(title)
        axes[0].set_xlabel('Time Index')
        axes[0].set_ylabel('Price (€/MWh)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Residuals
        for i, (model_name, pred) in enumerate(predictions.items()):
            if len(pred) == len(y_true):
                residuals = y_true.iloc[indices] - pred[indices]
                axes[1].plot(indices, residuals, label=f'{model_name} residuals', 
                           alpha=0.7, color=self.colors[i % len(self.colors)])
        
        axes[1].set_title('Prediction Residuals')
        axes[1].set_xlabel('Time Index')
        axes[1].set_ylabel('Residual (€/MWh)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def plot_error_distribution(self, y_true: pd.Series, predictions: Dict[str, np.ndarray],
                               title: str = "Error Distribution") -> None:
        """
        Plot error distribution for different models.
        
        Args:
            y_true: True values
            predictions: Dictionary of model predictions
            title: Plot title
        """
        n_models = len(predictions)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, pred) in enumerate(predictions.items()):
            if len(pred) == len(y_true):
                errors = y_true - pred
                axes[i].hist(errors, bins=30, alpha=0.7, density=True)
                axes[i].set_title(f'{model_name} Error Distribution')
                axes[i].set_xlabel('Error (€/MWh)')
                axes[i].set_ylabel('Density')
                axes[i].grid(True, alpha=0.3)
                
                # Add statistics
                mean_error = np.mean(errors)
                std_error = np.std(errors)
                axes[i].axvline(mean_error, color='red', linestyle='--', 
                              label=f'Mean: {mean_error:.2f}')
                axes[i].axvline(mean_error + std_error, color='orange', linestyle='--', 
                              label=f'±1σ: {std_error:.2f}')
                axes[i].axvline(mean_error - std_error, color='orange', linestyle='--')
                axes[i].legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def plot_metrics_comparison(self, results_df: pd.DataFrame, 
                               metrics: List[str] = None,
                               title: str = "Model Performance Comparison") -> None:
        """
        Plot comparison of different metrics across models.
        
        Args:
            results_df: DataFrame with model results
            metrics: List of metrics to plot
            title: Plot title
        """
        if metrics is None:
            metrics = ['rmse', 'mae', 'mape', 'directional_accuracy', 'r2']
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in results_df.columns]
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(5 * ((n_metrics + 1) // 2), 8))
        
        if n_metrics == 1:
            axes = [axes]
        elif n_metrics <= 2:
            axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics):
            ax = axes[i] if n_metrics > 1 else axes
            
            # Sort by metric value
            sorted_df = results_df.sort_values(metric, ascending=metric in ['rmse', 'mae', 'mape'])
            
            bars = ax.bar(range(len(sorted_df)), sorted_df[metric], 
                         color=self.colors[:len(sorted_df)])
            ax.set_title(f'{metric.upper()}')
            ax.set_xlabel('Models')
            ax.set_ylabel(metric.upper())
            ax.set_xticks(range(len(sorted_df)))
            ax.set_xticklabels(sorted_df.index, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Hide empty subplots
        if n_metrics < len(axes):
            for i in range(n_metrics, len(axes)):
                axes[i].set_visible(False)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def plot_rolling_metrics(self, rolling_metrics: pd.DataFrame,
                           metrics: List[str] = ['mae', 'rmse', 'mape'],
                           title: str = "Rolling Performance Metrics") -> None:
        """
        Plot rolling metrics over time.
        
        Args:
            rolling_metrics: DataFrame with rolling metrics
            metrics: List of metrics to plot
            title: Plot title
        """
        fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 4 * len(metrics)))
        
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric in rolling_metrics.columns:
                axes[i].plot(rolling_metrics.index, rolling_metrics[metric], 
                           linewidth=2, alpha=0.8)
                axes[i].set_title(f'Rolling {metric.upper()}')
                axes[i].set_xlabel('Time Index')
                axes[i].set_ylabel(metric.upper())
                axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def plot_price_level_accuracy(self, y_true: pd.Series, predictions: Dict[str, np.ndarray],
                                price_levels: List[float] = None,
                                title: str = "Accuracy by Price Level") -> None:
        """
        Plot accuracy metrics by price level.
        
        Args:
            y_true: True values
            predictions: Dictionary of model predictions
            price_levels: Price level thresholds
            title: Plot title
        """
        if price_levels is None:
            price_levels = [0, 25, 50, 75, 100, 150, 200, np.inf]
        
        n_models = len(predictions)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, pred) in enumerate(predictions.items()):
            if len(pred) == len(y_true):
                accuracy_by_level = []
                level_labels = []
                
                for j in range(len(price_levels) - 1):
                    lower = price_levels[j]
                    upper = price_levels[j + 1]
                    
                    mask = (y_true >= lower) & (y_true < upper)
                    if np.any(mask):
                        y_level = y_true[mask]
                        pred_level = pred[mask]
                        
                        mae = np.mean(np.abs(y_level - pred_level))
                        accuracy_by_level.append(mae)
                        level_labels.append(f'{lower}-{upper if upper != np.inf else "∞"}')
                
                axes[i].bar(range(len(accuracy_by_level)), accuracy_by_level, 
                          color=self.colors[i % len(self.colors)], alpha=0.7)
                axes[i].set_title(f'{model_name} - MAE by Price Level')
                axes[i].set_xlabel('Price Level (€/MWh)')
                axes[i].set_ylabel('MAE (€/MWh)')
                axes[i].set_xticks(range(len(level_labels)))
                axes[i].set_xticklabels(level_labels, rotation=45, ha='right')
                axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def plot_hourly_performance(self, y_true: pd.Series, predictions: Dict[str, np.ndarray],
                              title: str = "Performance by Hour of Day") -> None:
        """
        Plot performance metrics by hour of day.
        
        Args:
            y_true: True values
            predictions: Dictionary of model predictions
            title: Plot title
        """
        # Assume hourly data - extract hour from index
        if hasattr(y_true.index, 'hour'):
            hours = y_true.index.hour
        else:
            # If no datetime index, assume hourly data starting from hour 0
            hours = np.arange(len(y_true)) % 24
        
        n_models = len(predictions)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, pred) in enumerate(predictions.items()):
            if len(pred) == len(y_true):
                hourly_mae = []
                hourly_rmse = []
                
                for hour in range(24):
                    hour_mask = hours == hour
                    if np.any(hour_mask):
                        y_hour = y_true[hour_mask]
                        pred_hour = pred[hour_mask]
                        
                        mae = np.mean(np.abs(y_hour - pred_hour))
                        rmse = np.sqrt(np.mean((y_hour - pred_hour) ** 2))
                        
                        hourly_mae.append(mae)
                        hourly_rmse.append(rmse)
                    else:
                        hourly_mae.append(0)
                        hourly_rmse.append(0)
                
                x = np.arange(24)
                width = 0.35
                
                axes[i].bar(x - width/2, hourly_mae, width, label='MAE', alpha=0.7)
                axes[i].bar(x + width/2, hourly_rmse, width, label='RMSE', alpha=0.7)
                
                axes[i].set_title(f'{model_name} - Performance by Hour')
                axes[i].set_xlabel('Hour of Day')
                axes[i].set_ylabel('Error (€/MWh)')
                axes[i].set_xticks(x)
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def plot_scatter_predictions(self, y_true: pd.Series, predictions: Dict[str, np.ndarray],
                               title: str = "Predictions vs Actual Scatter") -> None:
        """
        Plot scatter plot of predictions vs actual values.
        
        Args:
            y_true: True values
            predictions: Dictionary of model predictions
            title: Plot title
        """
        n_models = len(predictions)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, pred) in enumerate(predictions.items()):
            if len(pred) == len(y_true):
                axes[i].scatter(y_true, pred, alpha=0.6, s=20)
                
                # Add perfect prediction line
                min_val = min(y_true.min(), pred.min())
                max_val = max(y_true.max(), pred.max())
                axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                
                # Calculate R²
                from sklearn.metrics import r2_score
                r2 = r2_score(y_true, pred)
                
                axes[i].set_title(f'{model_name} (R² = {r2:.3f})')
                axes[i].set_xlabel('Actual Price (€/MWh)')
                axes[i].set_ylabel('Predicted Price (€/MWh)')
                axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, feature_importance: Dict[str, pd.DataFrame],
                              top_n: int = 20,
                              title: str = "Feature Importance") -> None:
        """
        Plot feature importance for different models.
        
        Args:
            feature_importance: Dictionary of feature importance DataFrames
            top_n: Number of top features to plot
            title: Plot title
        """
        n_models = len(feature_importance)
        fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 6))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, importance_df) in enumerate(feature_importance.items()):
            top_features = importance_df.head(top_n)
            
            axes[i].barh(range(len(top_features)), top_features['importance'])
            axes[i].set_yticks(range(len(top_features)))
            axes[i].set_yticklabels(top_features['feature'])
            axes[i].set_title(f'{model_name.title()} - Top {top_n} Features')
            axes[i].set_xlabel('Importance')
            axes[i].invert_yaxis()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def create_dashboard(self, y_true: pd.Series, predictions: Dict[str, np.ndarray],
                        results_df: pd.DataFrame,
                        title: str = "Electricity Price Forecasting Dashboard") -> None:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            y_true: True values
            predictions: Dictionary of model predictions
            results_df: DataFrame with model results
            title: Dashboard title
        """
        fig = plt.figure(figsize=(20, 15))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Time series plot (top row, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        indices = np.linspace(0, len(y_true)-1, min(1000, len(y_true)), dtype=int)
        ax1.plot(indices, y_true.iloc[indices], label='Actual', linewidth=2, alpha=0.8)
        
        for i, (model_name, pred) in enumerate(predictions.items()):
            if len(pred) == len(y_true):
                ax1.plot(indices, pred[indices], label=model_name, alpha=0.7)
        
        ax1.set_title('Predictions vs Actual')
        ax1.set_xlabel('Time Index')
        ax1.set_ylabel('Price (€/MWh)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Metrics comparison (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        metrics = ['rmse', 'mae', 'mape']
        available_metrics = [m for m in metrics if m in results_df.columns]
        
        if available_metrics:
            sorted_df = results_df.sort_values(available_metrics[0])
            x = np.arange(len(sorted_df))
            width = 0.25
            
            for i, metric in enumerate(available_metrics):
                ax2.bar(x + i * width, sorted_df[metric], width, label=metric.upper())
            
            ax2.set_title('Model Comparison')
            ax2.set_xlabel('Models')
            ax2.set_ylabel('Error')
            ax2.set_xticks(x + width)
            ax2.set_xticklabels(sorted_df.index, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Error distribution (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        if predictions:
            first_model = list(predictions.keys())[0]
            pred = predictions[first_model]
            if len(pred) == len(y_true):
                errors = y_true - pred
                ax3.hist(errors, bins=30, alpha=0.7, density=True)
                ax3.set_title(f'Error Distribution - {first_model}')
                ax3.set_xlabel('Error (€/MWh)')
                ax3.set_ylabel('Density')
                ax3.grid(True, alpha=0.3)
        
        # 4. Scatter plot (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        if predictions:
            first_model = list(predictions.keys())[0]
            pred = predictions[first_model]
            if len(pred) == len(y_true):
                ax4.scatter(y_true, pred, alpha=0.6, s=20)
                min_val = min(y_true.min(), pred.min())
                max_val = max(y_true.max(), pred.max())
                ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                ax4.set_title(f'Scatter Plot - {first_model}')
                ax4.set_xlabel('Actual (€/MWh)')
                ax4.set_ylabel('Predicted (€/MWh)')
                ax4.grid(True, alpha=0.3)
        
        # 5. Hourly performance (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        if predictions:
            first_model = list(predictions.keys())[0]
            pred = predictions[first_model]
            if len(pred) == len(y_true):
                hours = np.arange(len(y_true)) % 24
                hourly_mae = []
                for hour in range(24):
                    hour_mask = hours == hour
                    if np.any(hour_mask):
                        mae = np.mean(np.abs(y_true[hour_mask] - pred[hour_mask]))
                        hourly_mae.append(mae)
                    else:
                        hourly_mae.append(0)
                
                ax5.bar(range(24), hourly_mae, alpha=0.7)
                ax5.set_title(f'Hourly MAE - {first_model}')
                ax5.set_xlabel('Hour of Day')
                ax5.set_ylabel('MAE (€/MWh)')
                ax5.grid(True, alpha=0.3)
        
        # 6. Summary statistics (bottom row, spans all columns)
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        # Create summary table
        summary_text = "Model Performance Summary\n\n"
        for model_name, pred in predictions.items():
            if len(pred) == len(y_true):
                mae = np.mean(np.abs(y_true - pred))
                rmse = np.sqrt(np.mean((y_true - pred) ** 2))
                mape = np.mean(np.abs((y_true - pred) / y_true)) * 100
                summary_text += f"{model_name}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%\n"
        
        ax6.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.suptitle(title, fontsize=16, y=0.98)
        plt.show()


def main():
    """Example usage of visualization tools."""
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic electricity price data
    t = np.arange(n_samples)
    price = 50 + 10 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 2, n_samples)
    
    # Create true and predicted values
    y_true = pd.Series(price, name='price')
    predictions = {
        'Model A': price + np.random.normal(0, 1, n_samples),
        'Model B': price + np.random.normal(0, 1.5, n_samples),
        'Model C': price + np.random.normal(0, 0.8, n_samples)
    }
    
    # Create results DataFrame
    results_data = []
    for model_name, pred in predictions.items():
        mae = np.mean(np.abs(y_true - pred))
        rmse = np.sqrt(np.mean((y_true - pred) ** 2))
        mape = np.mean(np.abs((y_true - pred) / y_true)) * 100
        r2 = 1 - np.sum((y_true - pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
        
        results_data.append({
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        })
    
    results_df = pd.DataFrame(results_data, index=list(predictions.keys()))
    
    # Create visualizations
    viz = ModelVisualization()
    
    # Plot predictions vs actual
    viz.plot_predictions_vs_actual(y_true, predictions)
    
    # Plot error distribution
    viz.plot_error_distribution(y_true, predictions)
    
    # Plot metrics comparison
    viz.plot_metrics_comparison(results_df)
    
    # Create dashboard
    viz.create_dashboard(y_true, predictions, results_df)


if __name__ == "__main__":
    main()
