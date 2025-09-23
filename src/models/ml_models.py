"""
Machine learning models for electricity price forecasting.

This module implements various machine learning models including
linear regression, tree-based models, and ensemble methods.
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. XGBoost models will be disabled.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. LightGBM models will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLModels:
    """
    Collection of machine learning models for electricity price forecasting.
    
    This class provides a unified interface for training and evaluating
    various ML models with proper preprocessing and hyperparameter tuning.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize ML models collection.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.scaler = None
        self.models = self._initialize_models()
        self.trained_models = {}
        self.results = {}
        self.feature_importance = {}
    
    def _initialize_models(self) -> Dict[str, BaseEstimator]:
        """Initialize all ML models with default parameters."""
        models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0, random_state=self.random_state),
            'lasso': Lasso(alpha=0.1, random_state=self.random_state),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state
            ),
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                random_state=self.random_state,
                max_iter=500
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        
        return models
    
    def _get_hyperparameter_grids(self) -> Dict[str, Dict[str, List]]:
        """Get hyperparameter grids for grid search."""
        grids = {
            'ridge': {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            'lasso': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
            },
            'elastic_net': {
                'alpha': [0.001, 0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 10]
            },
            'svr': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            },
            'mlp': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        }
        
        # Add XGBoost grid if available
        if XGBOOST_AVAILABLE:
            grids['xgboost'] = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 10],
                'subsample': [0.8, 0.9, 1.0]
            }
        
        # Add LightGBM grid if available
        if LIGHTGBM_AVAILABLE:
            grids['lightgbm'] = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 10],
                'subsample': [0.8, 0.9, 1.0]
            }
        
        return grids
    
    def preprocess_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                          scaler_type: str = 'standard') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess features using scaling and handle missing values.
        
        Args:
            X_train: Training features
            X_test: Test features
            scaler_type: Type of scaler ('standard', 'robust')
            
        Returns:
            Tuple of (scaled_X_train, scaled_X_test)
        """
        # Handle missing values first
        X_train_clean = X_train.fillna(X_train.mean())
        X_test_clean = X_test.fillna(X_train.mean())  # Use training mean for test data
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'robust'")
        
        # Fit scaler on training data
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train_clean),
            columns=X_train_clean.columns,
            index=X_train_clean.index
        )
        
        # Transform test data
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test_clean),
            columns=X_test_clean.columns,
            index=X_test_clean.index
        )
        
        logger.info(f"Features preprocessed using {scaler_type} scaling")
        return X_train_scaled, X_test_scaled
    
    def train_all(self, X_train: pd.DataFrame, y_train: pd.Series, 
                  tune_hyperparameters: bool = False, cv_folds: int = 5) -> Dict[str, BaseEstimator]:
        """
        Train all ML models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            tune_hyperparameters: Whether to perform hyperparameter tuning
            cv_folds: Number of CV folds for hyperparameter tuning
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Training ML models...")
        
        # Preprocess features
        X_train_scaled, _ = self.preprocess_features(X_train, X_train)
        
        # Get hyperparameter grids
        param_grids = self._get_hyperparameter_grids()
        
        # Time series split for cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}...")
                
                if tune_hyperparameters and name in param_grids:
                    # Hyperparameter tuning
                    grid_search = GridSearchCV(
                        model,
                        param_grids[name],
                        cv=tscv,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        verbose=0
                    )
                    grid_search.fit(X_train_scaled, y_train)
                    trained_model = grid_search.best_estimator_
                    logger.info(f"{name} best params: {grid_search.best_params_}")
                else:
                    # Train with default parameters
                    trained_model = model.fit(X_train_scaled, y_train)
                
                self.trained_models[name] = trained_model
                logger.info(f"{name} trained successfully")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        return self.trained_models
    
    def predict_all(self, X_test: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions with all trained models.
        
        Args:
            X_test: Test features
            
        Returns:
            Dictionary of predictions
        """
        if self.scaler is None:
            raise ValueError("Models must be trained first. Run train_all() before predict_all().")
        
        # Scale test features
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        predictions = {}
        
        for name, model in self.trained_models.items():
            try:
                pred = model.predict(X_test_scaled)
                predictions[name] = pred
                logger.info(f"Predictions generated for {name}")
            except Exception as e:
                logger.error(f"Error predicting with {name}: {e}")
        
        return predictions
    
    def evaluate_all(self, y_test: pd.Series, predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Evaluate all models using multiple metrics.
        
        Args:
            y_test: True test values
            predictions: Dictionary of predictions
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        for name, pred in predictions.items():
            if len(pred) != len(y_test):
                logger.warning(f"Length mismatch for {name}: pred={len(pred)}, test={len(y_test)}")
                continue
            
            # Calculate metrics
            mse = mean_squared_error(y_test, pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, pred)
            mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
            
            # Directional accuracy
            if len(pred) > 1:
                y_diff = np.diff(y_test.values)
                pred_diff = np.diff(pred)
                directional_accuracy = np.mean((y_diff * pred_diff) > 0) * 100
            else:
                directional_accuracy = 0
            
            results.append({
                'model': name,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'directional_accuracy': directional_accuracy
            })
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Get feature importance for tree-based models.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature importance DataFrames
        """
        importance_models = ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']
        
        for name in importance_models:
            if name in self.trained_models:
                try:
                    model = self.trained_models[name]
                    
                    if hasattr(model, 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            'feature': feature_names,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        self.feature_importance[name] = importance_df
                        logger.info(f"Feature importance calculated for {name}")
                
                except Exception as e:
                    logger.error(f"Error calculating feature importance for {name}: {e}")
        
        return self.feature_importance
    
    def get_best_model(self, metric: str = 'rmse') -> Tuple[str, BaseEstimator]:
        """
        Get the best performing model based on a specific metric.
        
        Args:
            metric: Metric to use for comparison ('rmse', 'mae', 'mape')
            
        Returns:
            Tuple of (model_name, model_instance)
        """
        if self.results.empty:
            raise ValueError("No evaluation results available. Run evaluate_all() first.")
        
        best_idx = self.results[metric].idxmin()
        best_model_name = self.results.loc[best_idx, 'model']
        best_model = self.trained_models[best_model_name]
        
        return best_model_name, best_model
    
    def get_model_summary(self) -> pd.DataFrame:
        """
        Get summary of all model results.
        
        Returns:
            DataFrame with model performance summary
        """
        if self.results.empty:
            raise ValueError("No evaluation results available. Run evaluate_all() first.")
        
        return self.results.sort_values('rmse').round(4)
    
    def plot_feature_importance(self, model_name: str, top_n: int = 20) -> None:
        """
        Plot feature importance for a specific model.
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to plot
        """
        import matplotlib.pyplot as plt
        
        if model_name not in self.feature_importance:
            logger.warning(f"No feature importance available for {model_name}")
            return
        
        importance_df = self.feature_importance[model_name].head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance - {model_name.title()}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, y_test: pd.Series, predictions: Dict[str, np.ndarray], 
                        title: str = "ML Model Predictions", 
                        max_plot_points: int = 1000) -> None:
        """
        Plot predictions vs actual values.
        
        Args:
            y_test: True test values
            predictions: Dictionary of predictions
            title: Plot title
            max_plot_points: Maximum number of points to plot
        """
        import matplotlib.pyplot as plt
        
        # Limit points for plotting
        plot_indices = np.linspace(0, len(y_test)-1, min(max_plot_points, len(y_test)), dtype=int)
        
        plt.figure(figsize=(15, 10))
        
        # Plot actual values
        plt.subplot(2, 1, 1)
        plt.plot(plot_indices, y_test.iloc[plot_indices], label='Actual', alpha=0.7, linewidth=2)
        
        # Plot predictions
        colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
        for i, (name, pred) in enumerate(predictions.items()):
            plt.plot(plot_indices, pred[plot_indices], label=name, alpha=0.7, color=colors[i])
        
        plt.title(title)
        plt.xlabel('Time Index')
        plt.ylabel('Price (€/MWh)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Plot residuals
        plt.subplot(2, 1, 2)
        for i, (name, pred) in enumerate(predictions.items()):
            residuals = y_test.iloc[plot_indices] - pred[plot_indices]
            plt.plot(plot_indices, residuals, label=f'{name} residuals', alpha=0.7, color=colors[i])
        
        plt.title('Prediction Residuals')
        plt.xlabel('Time Index')
        plt.ylabel('Residual (€/MWh)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save
            filepath: Path where to save the model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.trained_models.keys())}")
        
        model_data = {
            'model': self.trained_models[model_name],
            'scaler': self.scaler,
            'feature_importance': self.feature_importance.get(model_name),
            'results': self.results.get(model_name) if hasattr(self.results, 'get') else None
        }
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save using joblib for better compatibility with scikit-learn
        joblib.dump(model_data, filepath)
        logger.info(f"Model '{model_name}' saved to {filepath}")
    
    def load_model(self, model_name: str, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_name: Name to assign to the loaded model
            filepath: Path where the model is saved
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model data
        model_data = joblib.load(filepath)
        
        # Restore model and related data
        self.trained_models[model_name] = model_data['model']
        if model_data['scaler'] is not None:
            self.scaler = model_data['scaler']
        if model_data['feature_importance'] is not None:
            self.feature_importance[model_name] = model_data['feature_importance']
        if model_data['results'] is not None:
            if not hasattr(self, 'results') or self.results is None:
                self.results = {}
            self.results[model_name] = model_data['results']
        
        logger.info(f"Model '{model_name}' loaded from {filepath}")
    
    def save_all_models(self, directory: str) -> None:
        """
        Save all trained models to a directory.
        
        Args:
            directory: Directory where to save the models
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        for model_name in self.trained_models.keys():
            filepath = directory / f"{model_name}.pkl"
            self.save_model(model_name, str(filepath))
        
        logger.info(f"All models saved to {directory}")
    
    def load_all_models(self, directory: str) -> None:
        """
        Load all models from a directory.
        
        Args:
            directory: Directory where the models are saved
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        model_files = list(directory.glob("*.pkl"))
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {directory}")
        
        for model_file in model_files:
            model_name = model_file.stem
            self.load_model(model_name, str(model_file))
        
        logger.info(f"Loaded {len(model_files)} models from {directory}")


def main():
    """Example usage of ML models."""
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic electricity price data with trend and seasonality
    t = np.arange(n_samples)
    trend = 0.01 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 24)  # Daily seasonality
    noise = np.random.normal(0, 2, n_samples)
    price = 50 + trend + seasonal + noise
    
    # Create features
    features = pd.DataFrame({
        'hour': t % 24,
        'day_of_week': (t // 24) % 7,
        'temperature': 20 + 10 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 2, n_samples),
        'demand': 100 + 20 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 5, n_samples),
        'lag_1': np.roll(price, 1),
        'lag_24': np.roll(price, 24)
    })
    
    y = pd.Series(price, name='price')
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train and evaluate ML models
    ml_models = MLModels()
    ml_models.train_all(X_train, y_train, tune_hyperparameters=False)
    predictions = ml_models.predict_all(X_test)
    results = ml_models.evaluate_all(y_test, predictions)
    
    print("ML Model Results:")
    print(results)
    
    # Get feature importance
    feature_importance = ml_models.get_feature_importance(features.columns.tolist())
    print(f"\nFeature importance calculated for {len(feature_importance)} models")
    
    # Get best model
    best_name, best_model = ml_models.get_best_model('rmse')
    print(f"\nBest model: {best_name}")
    
    # Plot results
    ml_models.plot_predictions(y_test, predictions)


if __name__ == "__main__":
    main()
