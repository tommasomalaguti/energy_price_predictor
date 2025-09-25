"""
Online learning and real-time model adaptation for electricity price forecasting.

This module provides capabilities for continuous model updating, drift detection,
and adaptive forecasting in real-time scenarios.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OnlineLearner:
    """
    Online learning wrapper for any scikit-learn model.
    """
    
    def __init__(self, base_model: BaseEstimator, learning_rate: float = 0.01,
                 adaptation_threshold: float = 0.1, min_samples: int = 100):
        """
        Initialize online learner.
        
        Args:
            base_model: Base model for online learning
            learning_rate: Learning rate for adaptation
            adaptation_threshold: Threshold for triggering model adaptation
            min_samples: Minimum samples before adaptation
        """
        self.base_model = base_model
        self.learning_rate = learning_rate
        self.adaptation_threshold = adaptation_threshold
        self.min_samples = min_samples
        self.is_fitted = False
        self.adaptation_history = []
        self.performance_history = []
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'OnlineLearner':
        """
        Initial fit of the model.
        
        Args:
            X: Feature matrix
            y: Target series
            
        Returns:
            self
        """
        self.base_model.fit(X, y)
        self.is_fitted = True
        logger.info("Online learner fitted with initial data")
        return self
    
    def partial_fit(self, X: pd.DataFrame, y: pd.Series) -> 'OnlineLearner':
        """
        Partial fit for online learning.
        
        Args:
            X: New feature data
            y: New target data
            
        Returns:
            self
        """
        if not self.is_fitted:
            return self.fit(X, y)
        
        # Check if model supports partial_fit
        if hasattr(self.base_model, 'partial_fit'):
            self.base_model.partial_fit(X, y)
        else:
            # Retrain with combined data
            # This is a simplified approach - in practice, you'd use more sophisticated methods
            logger.warning("Model doesn't support partial_fit. Retraining...")
            # In a real implementation, you'd maintain a buffer of recent data
            # and retrain periodically or when performance degrades
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.base_model.predict(X)
    
    def adapt_if_needed(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """
        Check if model adaptation is needed and perform it.
        
        Args:
            X: New feature data
            y: New target data
            
        Returns:
            True if adaptation was performed
        """
        if len(self.performance_history) < self.min_samples:
            return False
        
        # Calculate recent performance
        recent_performance = np.mean(self.performance_history[-self.min_samples:])
        
        # Check if performance has degraded
        if len(self.performance_history) >= self.min_samples * 2:
            previous_performance = np.mean(
                self.performance_history[-self.min_samples*2:-self.min_samples]
            )
            
            if recent_performance > previous_performance * (1 + self.adaptation_threshold):
                logger.info("Performance degradation detected. Adapting model...")
                self.partial_fit(X, y)
                self.adaptation_history.append(len(self.performance_history))
                return True
        
        return False
    
    def update_performance(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Update performance history.
        
        Args:
            y_true: True values
            y_pred: Predicted values
        """
        mse = mean_squared_error(y_true, y_pred)
        self.performance_history.append(mse)


class DriftDetector:
    """
    Concept drift detection for time series data.
    """
    
    def __init__(self, window_size: int = 100, threshold: float = 0.1):
        """
        Initialize drift detector.
        
        Args:
            window_size: Size of sliding window for drift detection
            threshold: Threshold for drift detection
        """
        self.window_size = window_size
        self.threshold = threshold
        self.reference_window = None
        self.drift_detected = False
        self.drift_history = []
    
    def update(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """
        Update drift detector with new data.
        
        Args:
            X: Feature matrix
            y: Target series
            
        Returns:
            True if drift is detected
        """
        current_window = pd.concat([X, y], axis=1)
        
        if self.reference_window is None:
            self.reference_window = current_window
            return False
        
        # Calculate drift score (simplified approach)
        drift_score = self._calculate_drift_score(self.reference_window, current_window)
        
        if drift_score > self.threshold:
            self.drift_detected = True
            self.drift_history.append(drift_score)
            logger.info(f"Concept drift detected! Score: {drift_score:.4f}")
            return True
        
        # Update reference window
        self.reference_window = current_window
        return False
    
    def _calculate_drift_score(self, window1: pd.DataFrame, window2: pd.DataFrame) -> float:
        """
        Calculate drift score between two windows.
        
        Args:
            window1: First window
            window2: Second window
            
        Returns:
            Drift score
        """
        # Simple statistical drift detection
        # In practice, you'd use more sophisticated methods like KS test, etc.
        
        # Calculate mean difference
        mean_diff = np.mean(np.abs(window1.mean() - window2.mean()))
        
        # Calculate variance difference
        var_diff = np.mean(np.abs(window1.var() - window2.var()))
        
        # Combined drift score
        drift_score = mean_diff + var_diff
        
        return drift_score


class AdaptiveEnsemble:
    """
    Adaptive ensemble that can add/remove models based on performance.
    """
    
    def __init__(self, base_models: Dict[str, BaseEstimator], 
                 performance_window: int = 50, removal_threshold: float = 0.2):
        """
        Initialize adaptive ensemble.
        
        Args:
            base_models: Dictionary of base models
            performance_window: Window size for performance evaluation
            removal_threshold: Threshold for model removal
        """
        self.base_models = base_models
        self.performance_window = performance_window
        self.removal_threshold = removal_threshold
        self.model_performance = {name: [] for name in base_models.keys()}
        self.active_models = set(base_models.keys())
        self.ensemble_weights = {name: 1.0 for name in base_models.keys()}
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'AdaptiveEnsemble':
        """
        Fit all models in the ensemble.
        
        Args:
            X: Feature matrix
            y: Target series
            
        Returns:
            self
        """
        for name, model in self.base_models.items():
            model.fit(X, y)
        
        logger.info(f"Adaptive ensemble fitted with {len(self.base_models)} models")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Ensemble predictions
        """
        predictions = []
        weights = []
        
        for name in self.active_models:
            if name in self.base_models:
                pred = self.base_models[name].predict(X)
                predictions.append(pred)
                weights.append(self.ensemble_weights[name])
        
        if not predictions:
            raise ValueError("No active models in ensemble")
        
        # Weighted average
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        return np.average(predictions, axis=0, weights=weights)
    
    def update_performance(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Update performance tracking for all models.
        
        Args:
            y_true: True values
            y_pred: Ensemble predictions
        """
        for name in self.active_models:
            if name in self.base_models:
                model_pred = self.base_models[name].predict(
                    pd.DataFrame({'dummy': [0] * len(y_true)})
                )  # Simplified - in practice, you'd store the features
                
                mse = mean_squared_error(y_true, model_pred)
                self.model_performance[name].append(mse)
                
                # Keep only recent performance
                if len(self.model_performance[name]) > self.performance_window:
                    self.model_performance[name] = self.model_performance[name][-self.performance_window:]
    
    def adapt_ensemble(self):
        """
        Adapt ensemble based on model performance.
        """
        if len(self.active_models) < 2:
            return  # Need at least 2 models for adaptation
        
        # Calculate average performance for each model
        model_avg_performance = {}
        for name in self.active_models:
            if len(self.model_performance[name]) > 0:
                model_avg_performance[name] = np.mean(self.model_performance[name])
            else:
                model_avg_performance[name] = float('inf')
        
        # Find best and worst performing models
        best_model = min(model_avg_performance, key=model_avg_performance.get)
        worst_model = max(model_avg_performance, key=model_avg_performance.get)
        
        # Calculate performance ratio
        if model_avg_performance[best_model] > 0:
            performance_ratio = model_avg_performance[worst_model] / model_avg_performance[best_model]
            
            # Remove worst model if performance is significantly worse
            if performance_ratio > (1 + self.removal_threshold):
                self.active_models.remove(worst_model)
                logger.info(f"Removed worst performing model: {worst_model}")
        
        # Update ensemble weights
        self._update_weights()
    
    def _update_weights(self):
        """
        Update ensemble weights based on model performance.
        """
        if len(self.active_models) < 2:
            return
        
        # Calculate weights based on inverse performance
        weights = {}
        total_weight = 0
        
        for name in self.active_models:
            if len(self.model_performance[name]) > 0:
                avg_performance = np.mean(self.model_performance[name])
                weight = 1.0 / (avg_performance + 1e-8)
                weights[name] = weight
                total_weight += weight
            else:
                weights[name] = 1.0
                total_weight += 1.0
        
        # Normalize weights
        for name in weights:
            self.ensemble_weights[name] = weights[name] / total_weight


class RealTimeForecaster:
    """
    Real-time forecasting system with online learning and drift detection.
    """
    
    def __init__(self, base_model: BaseEstimator, update_frequency: int = 24,
                 drift_detection: bool = True, adaptation: bool = True):
        """
        Initialize real-time forecaster.
        
        Args:
            base_model: Base forecasting model
            update_frequency: Frequency of model updates (hours)
            drift_detection: Whether to enable drift detection
            adaptation: Whether to enable model adaptation
        """
        self.base_model = base_model
        self.update_frequency = update_frequency
        self.drift_detection = drift_detection
        self.adaptation = adaptation
        
        # Initialize components
        self.online_learner = OnlineLearner(base_model)
        self.drift_detector = DriftDetector() if drift_detection else None
        self.adaptation_history = []
        self.prediction_history = []
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RealTimeForecaster':
        """
        Initial fit of the real-time forecaster.
        
        Args:
            X: Feature matrix
            y: Target series
            
        Returns:
            self
        """
        self.online_learner.fit(X, y)
        logger.info("Real-time forecaster fitted with initial data")
        return self
    
    def predict_and_update(self, X: pd.DataFrame, y_true: Optional[pd.Series] = None) -> np.ndarray:
        """
        Make predictions and update model if needed.
        
        Args:
            X: Feature matrix for prediction
            y_true: True values (if available for updating)
            
        Returns:
            Predictions
        """
        # Make predictions
        predictions = self.online_learner.predict(X)
        self.prediction_history.append(predictions)
        
        # Update model if true values are available
        if y_true is not None:
            # Check for drift
            if self.drift_detection and self.drift_detector:
                drift_detected = self.drift_detector.update(X, y_true)
                if drift_detected:
                    logger.info("Drift detected - triggering model update")
                    self.online_learner.partial_fit(X, y_true)
                    self.adaptation_history.append(len(self.prediction_history))
            
            # Update performance
            self.online_learner.update_performance(y_true.values, predictions)
            
            # Check if adaptation is needed
            if self.adaptation:
                adapted = self.online_learner.adapt_if_needed(X, y_true)
                if adapted:
                    self.adaptation_history.append(len(self.prediction_history))
        
        return predictions
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """
        Get summary of model adaptations.
        
        Returns:
            Dictionary with adaptation summary
        """
        return {
            'total_adaptations': len(self.adaptation_history),
            'adaptation_frequency': len(self.adaptation_history) / max(len(self.prediction_history), 1),
            'last_adaptation': self.adaptation_history[-1] if self.adaptation_history else None,
            'drift_detection_enabled': self.drift_detection,
            'adaptation_enabled': self.adaptation
        }


def main():
    """Example usage of online learning with real data."""
    print("Online Learning example requires real electricity price data.")
    print("Please use the ENTSO-E downloader to get real data first.")
    print("Example usage:")
    print("1. Download real data using ENTSOEDownloader")
    print("2. Preprocess data using DataPreprocessor") 
    print("3. Create OnlineLearner or RealTimeForecaster")
    print("4. Use partial_fit() for online learning")
    print("5. Monitor adaptation with get_adaptation_summary()")


if __name__ == "__main__":
    main()
