"""
Ensemble methods for electricity price forecasting.

This module implements various ensemble techniques including voting,
stacking, and bagging for improved prediction accuracy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from sklearn.ensemble import VotingRegressor, BaggingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
import joblib
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleModels:
    """
    Ensemble methods for electricity price forecasting.
    
    This class provides various ensemble techniques including voting,
    stacking, and bagging for improved prediction accuracy.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize ensemble models.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.ensemble_models = {}
        self.meta_models = {}
        self.results = {}
        self.feature_importance = {}
    
    def create_voting_ensemble(self, base_models: Dict[str, BaseEstimator], 
                               weights: Optional[List[float]] = None) -> VotingRegressor:
        """
        Create a voting ensemble from base models.
        
        Args:
            base_models: Dictionary of base models
            weights: Optional weights for each model
            
        Returns:
            VotingRegressor ensemble
        """
        estimators = [(name, model) for name, model in base_models.items()]
        
        voting_ensemble = VotingRegressor(
            estimators=estimators,
            weights=weights,
            voting='soft'  # Use average of predictions
        )
        
        self.ensemble_models['voting'] = voting_ensemble
        logger.info(f"Created voting ensemble with {len(base_models)} base models")
        return voting_ensemble
    
    def create_stacking_ensemble(self, base_models: Dict[str, BaseEstimator], 
                                meta_model: Optional[BaseEstimator] = None,
                                cv_folds: int = 5) -> 'StackingRegressor':
        """
        Create a stacking ensemble from base models.
        
        Args:
            base_models: Dictionary of base models
            meta_model: Meta-model for stacking (default: LinearRegression)
            cv_folds: Number of CV folds for stacking
            
        Returns:
            StackingRegressor ensemble
        """
        if meta_model is None:
            meta_model = LinearRegression()
        
        estimators = [(name, model) for name, model in base_models.items()]
        
        stacking_ensemble = StackingRegressor(
            estimators=estimators,
            final_estimator=meta_model,
            cv=cv_folds,
            n_jobs=-1
        )
        
        self.ensemble_models['stacking'] = stacking_ensemble
        self.meta_models['stacking'] = meta_model
        logger.info(f"Created stacking ensemble with {len(base_models)} base models")
        return stacking_ensemble
    
    def create_bagging_ensemble(self, base_model: BaseEstimator,
                               n_estimators: int = 10,
                               max_samples: float = 1.0,
                               max_features: float = 1.0) -> BaggingRegressor:
        """
        Create a bagging ensemble from a base model.
        
        Args:
            base_model: Base model to bag
            n_estimators: Number of estimators
            max_samples: Fraction of samples to use
            max_features: Fraction of features to use
            
        Returns:
            BaggingRegressor ensemble
        """
        bagging_ensemble = BaggingRegressor(
            base_estimator=base_model,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.ensemble_models['bagging'] = bagging_ensemble
        logger.info(f"Created bagging ensemble with {n_estimators} estimators")
        return bagging_ensemble
    
    def create_dynamic_ensemble(self, base_models: Dict[str, BaseEstimator], 
                                selection_method: str = 'performance') -> 'DynamicEnsemble':
        """
        Create a dynamic ensemble that selects models based on performance.
        
        Args:
            base_models: Dictionary of base models
            selection_method: Method for model selection ('performance', 'diversity', 'hybrid')
            
        Returns:
            DynamicEnsemble
        """
        dynamic_ensemble = DynamicEnsemble(
            base_models=base_models,
            selection_method=selection_method,
            random_state=self.random_state
        )
        
        self.ensemble_models['dynamic'] = dynamic_ensemble
        logger.info(f"Created dynamic ensemble with {len(base_models)} base models")
        return dynamic_ensemble
    
    def create_weighted_ensemble(self, base_models: Dict[str, BaseEstimator], 
                                 performance_weights: Optional[Dict[str, float]] = None) -> 'WeightedEnsemble':
        """
        Create a weighted ensemble based on model performance.
        
        Args:
            base_models: Dictionary of base models
            performance_weights: Optional performance-based weights
            
        Returns:
            WeightedEnsemble
        """
        weighted_ensemble = WeightedEnsemble(
            base_models=base_models,
            performance_weights=performance_weights,
            random_state=self.random_state
        )
        
        self.ensemble_models['weighted'] = weighted_ensemble
        logger.info(f"Created weighted ensemble with {len(base_models)} base models")
        return weighted_ensemble
    
    def train_ensemble(self, ensemble_name: str, X_train: pd.DataFrame, 
                      y_train: pd.Series) -> BaseEstimator:
        """
        Train an ensemble model.
        
        Args:
            ensemble_name: Name of the ensemble
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained ensemble model
        """
        if ensemble_name not in self.ensemble_models:
            raise ValueError(f"Ensemble '{ensemble_name}' not found")
        
        ensemble = self.ensemble_models[ensemble_name]
        ensemble.fit(X_train, y_train)
        
        logger.info(f"Trained {ensemble_name} ensemble")
        return ensemble
    
    def predict_ensemble(self, ensemble_name: str, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with an ensemble model.
        
        Args:
            ensemble_name: Name of the ensemble
            X_test: Test features
            
        Returns:
            Predictions
        """
        if ensemble_name not in self.ensemble_models:
            raise ValueError(f"Ensemble '{ensemble_name}' not found")
        
        ensemble = self.ensemble_models[ensemble_name]
        predictions = ensemble.predict(X_test)
        
        logger.info(f"Generated predictions with {ensemble_name} ensemble")
        return predictions
    
    def evaluate_ensemble(self, ensemble_name: str, y_test: pd.Series, 
                         predictions: np.ndarray) -> Dict[str, float]:
        """
        Evaluate an ensemble model.
        
        Args:
            ensemble_name: Name of the ensemble
            y_test: True test values
            predictions: Predictions
            
        Returns:
            Dictionary of evaluation metrics
        """
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        r2 = r2_score(y_test, predictions)
        
        # Directional accuracy
        if len(predictions) > 1:
            y_diff = np.diff(y_test.values)
            pred_diff = np.diff(predictions)
            directional_accuracy = np.mean((y_diff * pred_diff) > 0) * 100
        else:
            directional_accuracy = 0
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'directional_accuracy': directional_accuracy
        }
        
        self.results[ensemble_name] = metrics
        logger.info(f"Evaluated {ensemble_name} ensemble: RMSE={rmse:.4f}, R²={r2:.4f}")
        return metrics
    
    def get_ensemble_weights(self, ensemble_name: str) -> Optional[np.ndarray]:
        """
        Get ensemble weights for voting ensembles.
        
        Args:
            ensemble_name: Name of the ensemble
            
        Returns:
            Ensemble weights or None
        """
        if ensemble_name not in self.ensemble_models:
            return None
        
        ensemble = self.ensemble_models[ensemble_name]
        
        if hasattr(ensemble, 'weights_'):
            return ensemble.weights_
        elif hasattr(ensemble, 'estimators_'):
            # For stacking, return meta-model coefficients
            if hasattr(ensemble, 'final_estimator_'):
                if hasattr(ensemble.final_estimator_, 'coef_'):
                    return ensemble.final_estimator_.coef_
        
        return None
    
    def get_base_model_predictions(self, ensemble_name: str, X_test: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get predictions from individual base models in an ensemble.
        
        Args:
            ensemble_name: Name of the ensemble
            X_test: Test features
            
        Returns:
            Dictionary of base model predictions
        """
        if ensemble_name not in self.ensemble_models:
            raise ValueError(f"Ensemble '{ensemble_name}' not found")
        
        ensemble = self.ensemble_models[ensemble_name]
        base_predictions = {}
        
        if hasattr(ensemble, 'estimators_'):
            for name, estimator in ensemble.estimators_:
                pred = estimator.predict(X_test)
                base_predictions[name] = pred
        
        return base_predictions
    
    def cross_validate_ensemble(self, ensemble_name: str, X: pd.DataFrame, 
                               y: pd.Series, cv_folds: int = 5) -> Dict[str, float]:
        """
        Cross-validate an ensemble model.
        
        Args:
            ensemble_name: Name of the ensemble
            X: Features
            y: Targets
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary of CV scores
        """
        if ensemble_name not in self.ensemble_models:
            raise ValueError(f"Ensemble '{ensemble_name}' not found")
        
        ensemble = self.ensemble_models[ensemble_name]
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Cross-validate with multiple metrics
        mse_scores = cross_val_score(ensemble, X, y, cv=kf, scoring='neg_mean_squared_error')
        mae_scores = cross_val_score(ensemble, X, y, cv=kf, scoring='neg_mean_absolute_error')
        r2_scores = cross_val_score(ensemble, X, y, cv=kf, scoring='r2')
        
        cv_results = {
            'mse_mean': -mse_scores.mean(),
            'mse_std': mse_scores.std(),
            'mae_mean': -mae_scores.mean(),
            'mae_std': mae_scores.std(),
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std()
        }
        
        logger.info(f"Cross-validated {ensemble_name} ensemble: R²={cv_results['r2_mean']:.4f}±{cv_results['r2_std']:.4f}")
        return cv_results
    
    def save_ensemble(self, ensemble_name: str, filepath: str) -> None:
        """
        Save an ensemble model to disk.
        
        Args:
            ensemble_name: Name of the ensemble
            filepath: Path where to save the model
        """
        if ensemble_name not in self.ensemble_models:
            raise ValueError(f"Ensemble '{ensemble_name}' not found")
        
        ensemble = self.ensemble_models[ensemble_name]
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save using joblib
        joblib.dump(ensemble, filepath)
        logger.info(f"Ensemble '{ensemble_name}' saved to {filepath}")
    
    def load_ensemble(self, ensemble_name: str, filepath: str) -> None:
        """
        Load an ensemble model from disk.
        
        Args:
            ensemble_name: Name to assign to the loaded ensemble
            filepath: Path where the model is saved
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Ensemble file not found: {filepath}")
        
        ensemble = joblib.load(filepath)
        self.ensemble_models[ensemble_name] = ensemble
        logger.info(f"Ensemble '{ensemble_name}' loaded from {filepath}")
    
    def get_ensemble_summary(self) -> pd.DataFrame:
        """
        Get summary of all ensemble results.
        
        Returns:
            DataFrame with ensemble performance summary
        """
        if not self.results:
            raise ValueError("No ensemble results available. Run evaluate_ensemble() first.")
        
        results_df = pd.DataFrame(self.results).T
        return results_df.sort_values('rmse').round(4)


class StackingRegressor(BaseEstimator, RegressorMixin):
    """
    Custom stacking regressor implementation.
    """
    
    def __init__(self, estimators, final_estimator=None, cv=5, n_jobs=-1):
        self.estimators = estimators
        self.final_estimator = final_estimator or LinearRegression()
        self.cv = cv
        self.n_jobs = n_jobs
    
    def fit(self, X, y):
        """Fit the stacking regressor."""
        from sklearn.model_selection import cross_val_predict
        
        # Get predictions from base models using cross-validation
        base_predictions = []
        self.estimators_ = []
        
        for name, estimator in self.estimators:
            # Fit the estimator
            estimator.fit(X, y)
            self.estimators_.append((name, estimator))
            
            # Get cross-validated predictions
            pred = cross_val_predict(estimator, X, y, cv=self.cv, n_jobs=self.n_jobs)
            base_predictions.append(pred)
        
        # Create meta-features
        meta_features = np.column_stack(base_predictions)
        
        # Fit the final estimator
        self.final_estimator_ = self.final_estimator
        self.final_estimator_.fit(meta_features, y)
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        # Get predictions from base models
        base_predictions = []
        for name, estimator in self.estimators_:
            pred = estimator.predict(X)
            base_predictions.append(pred)
        
        # Create meta-features
        meta_features = np.column_stack(base_predictions)
        
        # Make final prediction
        return self.final_estimator_.predict(meta_features)


class DynamicEnsemble(BaseEstimator, RegressorMixin):
    """
    Dynamic ensemble that selects models based on performance or diversity.
    """
    
    def __init__(self, base_models: Dict[str, BaseEstimator], 
                 selection_method: str = 'performance', 
                 n_models: int = 3,
                 random_state: int = 42):
        self.base_models = base_models
        self.selection_method = selection_method
        self.n_models = n_models
        self.random_state = random_state
        self.selected_models = {}
        self.model_performance = {}
        
    def fit(self, X, y):
        """Fit the dynamic ensemble."""
        # Evaluate all base models
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import mean_squared_error
        
        for name, model in self.base_models.items():
            try:
                # Cross-validation score
                scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
                self.model_performance[name] = -scores.mean()
            except:
                self.model_performance[name] = np.inf
        
        # Select top models based on performance
        if self.selection_method == 'performance':
            sorted_models = sorted(self.model_performance.items(), key=lambda x: x[1])
            selected_names = [name for name, _ in sorted_models[:self.n_models]]
        else:
            # Random selection for diversity
            np.random.seed(self.random_state)
            selected_names = np.random.choice(list(self.base_models.keys()), 
                                           size=min(self.n_models, len(self.base_models)), 
                                           replace=False)
        
        # Train selected models
        for name in selected_names:
            model = self.base_models[name]
            model.fit(X, y)
            self.selected_models[name] = model
        
        return self
    
    def predict(self, X):
        """Make predictions using selected models."""
        predictions = []
        weights = []
        
        for name, model in self.selected_models.items():
            pred = model.predict(X)
            predictions.append(pred)
            # Weight by inverse performance (lower error = higher weight)
            weight = 1.0 / (self.model_performance[name] + 1e-8)
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted average
        final_pred = np.zeros(len(X))
        for pred, weight in zip(predictions, weights):
            final_pred += weight * pred
        
        return final_pred


class WeightedEnsemble(BaseEstimator, RegressorMixin):
    """
    Weighted ensemble based on model performance.
    """
    
    def __init__(self, base_models: Dict[str, BaseEstimator], 
                 performance_weights: Optional[Dict[str, float]] = None,
                 random_state: int = 42):
        self.base_models = base_models
        self.performance_weights = performance_weights
        self.random_state = random_state
        self.trained_models = {}
        self.weights = {}
        
    def fit(self, X, y):
        """Fit the weighted ensemble."""
        # Train all models
        for name, model in self.base_models.items():
            model.fit(X, y)
            self.trained_models[name] = model
        
        # Calculate weights based on performance if not provided
        if self.performance_weights is None:
            from sklearn.model_selection import cross_val_score
            
            performance_scores = {}
            for name, model in self.trained_models.items():
                try:
                    scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
                    performance_scores[name] = -scores.mean()
                except:
                    performance_scores[name] = np.inf
            
            # Convert to weights (inverse of error)
            total_performance = sum(1.0 / (score + 1e-8) for score in performance_scores.values())
            for name, score in performance_scores.items():
                self.weights[name] = (1.0 / (score + 1e-8)) / total_performance
        else:
            self.weights = self.performance_weights
        
        return self
    
    def predict(self, X):
        """Make predictions using weighted ensemble."""
        predictions = []
        weights = []
        
        for name, model in self.trained_models.items():
            pred = model.predict(X)
            predictions.append(pred)
            weights.append(self.weights[name])
        
        # Weighted average
        final_pred = np.zeros(len(X))
        for pred, weight in zip(predictions, weights):
            final_pred += weight * pred
        
        return final_pred


def main():
    """Example usage of ensemble models with real data."""
    print("Ensemble Models example requires real electricity price data.")
    print("Please use the ENTSO-E downloader to get real data first.")
    print("Example usage:")
    print("1. Download real data using ENTSOEDownloader")
    print("2. Preprocess data using DataPreprocessor") 
    print("3. Train base models using MLModels")
    print("4. Create ensemble using EnsembleModels")
    print("5. Evaluate ensemble performance")


if __name__ == "__main__":
    main()
