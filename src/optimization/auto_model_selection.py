"""
Automated model selection and hyperparameter optimization.

This module provides automated model selection, hyperparameter tuning,
and model performance evaluation for electricity price forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Install with: pip install optuna")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class AutoModelSelector:
    """
    Automated model selection and hyperparameter optimization.
    """
    
    def __init__(self, cv_folds: int = 5, n_trials: int = 100, 
                 timeout: Optional[int] = None, random_state: int = 42):
        """
        Initialize auto model selector.
        
        Args:
            cv_folds: Number of cross-validation folds
            n_trials: Number of optimization trials
            timeout: Maximum time in seconds for optimization
            random_state: Random state for reproducibility
        """
        self.cv_folds = cv_folds
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_state = random_state
        self.best_models = {}
        self.optimization_results = {}
        
        # Initialize available models
        self.models = self._initialize_models()
    
    def _initialize_models(self) -> Dict[str, Dict]:
        """Initialize available models and their parameter spaces."""
        models = {
            'linear_regression': {
                'model': LinearRegression,
                'params': {}
            },
            'ridge': {
                'model': Ridge,
                'params': {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
                }
            },
            'lasso': {
                'model': Lasso,
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
                }
            },
            'random_forest': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 6, 10]
                }
            },
            'svr': {
                'model': SVR,
                'params': {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'kernel': ['rbf', 'poly', 'sigmoid']
                }
            },
            'mlp': {
                'model': MLPRegressor,
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['xgboost'] = {
                'model': xgb.XGBRegressor,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 6, 10],
                    'subsample': [0.8, 0.9, 1.0]
                }
            }
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = {
                'model': lgb.LGBMRegressor,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 6, 10],
                    'subsample': [0.8, 0.9, 1.0]
                }
            }
        
        # Add CatBoost if available
        if CATBOOST_AVAILABLE:
            models['catboost'] = {
                'model': cb.CatBoostRegressor,
                'params': {
                    'iterations': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'depth': [3, 6, 10],
                    'l2_leaf_reg': [1, 3, 5, 7, 9]
                }
            }
        
        return models
    
    def optimize_model(self, model_name: str, X_train: pd.DataFrame, 
                      y_train: pd.Series, scoring: str = 'neg_mean_squared_error') -> Dict[str, Any]:
        """
        Optimize hyperparameters for a specific model.
        
        Args:
            model_name: Name of the model to optimize
            X_train: Training features
            y_train: Training targets
            scoring: Scoring metric
            
        Returns:
            Dictionary with optimization results
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available. Using default parameters.")
            return self._optimize_without_optuna(model_name, X_train, y_train, scoring)
        
        model_config = self.models[model_name]
        model_class = model_config['model']
        param_space = model_config['params']
        
        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_values in param_space.items():
                if isinstance(param_values[0], (int, float)):
                    if param_name in ['alpha', 'learning_rate', 'C', 'gamma']:
                        params[param_name] = trial.suggest_float(param_name, 
                                                               min(param_values), 
                                                               max(param_values), 
                                                               log=True)
                    else:
                        params[param_name] = trial.suggest_int(param_name, 
                                                             min(param_values), 
                                                             max(param_values))
                elif isinstance(param_values[0], str):
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
                elif isinstance(param_values[0], tuple):
                    # Handle hidden_layer_sizes
                    n_layers = trial.suggest_int(f'{param_name}_layers', 1, 3)
                    layer_sizes = []
                    for i in range(n_layers):
                        layer_sizes.append(trial.suggest_int(f'{param_name}_size_{i}', 10, 200))
                    params[param_name] = tuple(layer_sizes)
            
            # Add random state
            if 'random_state' in model_class().get_params():
                params['random_state'] = self.random_state
            
            # Create and train model
            model = model_class(**params)
            
            # Cross-validation
            cv = TimeSeriesSplit(n_splits=self.cv_folds)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
            
            return scores.mean()
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner()
        )
        
        # Optimize
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        # Store results
        self.optimization_results[model_name] = {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
        
        # Train best model
        best_model = model_class(**study.best_params)
        best_model.fit(X_train, y_train)
        self.best_models[model_name] = best_model
        
        logger.info(f"Optimized {model_name}: score={study.best_value:.4f}")
        return self.optimization_results[model_name]
    
    def _optimize_without_optuna(self, model_name: str, X_train: pd.DataFrame, 
                                y_train: pd.Series, scoring: str) -> Dict[str, Any]:
        """Optimize model without Optuna using grid search."""
        from sklearn.model_selection import GridSearchCV
        
        model_config = self.models[model_name]
        model_class = model_config['model']
        param_grid = model_config['params']
        
        # Create model
        model = model_class()
        
        # Grid search
        cv = TimeSeriesSplit(n_splits=self.cv_folds)
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # Store results
        self.optimization_results[model_name] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'n_trials': len(grid_search.cv_results_['params'])
        }
        
        self.best_models[model_name] = grid_search.best_estimator_
        
        return self.optimization_results[model_name]
    
    def optimize_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                           models_to_optimize: Optional[List[str]] = None,
                           scoring: str = 'neg_mean_squared_error') -> Dict[str, Dict[str, Any]]:
        """
        Optimize all available models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            models_to_optimize: List of models to optimize (if None, optimizes all)
            scoring: Scoring metric
            
        Returns:
            Dictionary with optimization results for all models
        """
        if models_to_optimize is None:
            models_to_optimize = list(self.models.keys())
        
        logger.info(f"Optimizing {len(models_to_optimize)} models...")
        
        for model_name in models_to_optimize:
            try:
                logger.info(f"Optimizing {model_name}...")
                self.optimize_model(model_name, X_train, y_train, scoring)
            except Exception as e:
                logger.error(f"Error optimizing {model_name}: {e}")
                continue
        
        return self.optimization_results
    
    def get_best_model(self, metric: str = 'best_score') -> Tuple[str, Any]:
        """
        Get the best performing model.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model_name, model_instance)
        """
        if not self.optimization_results:
            raise ValueError("No optimization results available. Run optimize_all_models() first.")
        
        best_model_name = max(self.optimization_results.keys(), 
                             key=lambda x: self.optimization_results[x][metric])
        best_model = self.best_models[best_model_name]
        
        return best_model_name, best_model
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Evaluate all optimized models on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        for model_name, model in self.best_models.items():
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                r2 = r2_score(y_test, y_pred)
                
                # Directional accuracy
                if len(y_pred) > 1:
                    y_diff = np.diff(y_test.values)
                    pred_diff = np.diff(y_pred)
                    directional_accuracy = np.mean((y_diff * pred_diff) > 0) * 100
                else:
                    directional_accuracy = 0
                
                results.append({
                    'model': model_name,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape,
                    'r2': r2,
                    'directional_accuracy': directional_accuracy,
                    'cv_score': self.optimization_results[model_name]['best_score']
                })
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                continue
        
        return pd.DataFrame(results).sort_values('rmse')
    
    def plot_optimization_history(self, model_name: str) -> None:
        """
        Plot optimization history for a specific model.
        
        Args:
            model_name: Name of the model
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available for plotting optimization history")
            return
        
        if model_name not in self.optimization_results:
            logger.warning(f"No optimization results for {model_name}")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # This would require storing the study object
            logger.info(f"Optimization history for {model_name}: "
                       f"Best score: {self.optimization_results[model_name]['best_score']:.4f}")
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
    
    def get_model_ranking(self, metric: str = 'rmse') -> pd.DataFrame:
        """
        Get ranking of all models based on a specific metric.
        
        Args:
            metric: Metric to use for ranking
            
        Returns:
            DataFrame with model ranking
        """
        if not self.optimization_results:
            raise ValueError("No optimization results available")
        
        ranking_data = []
        for model_name, results in self.optimization_results.items():
            ranking_data.append({
                'model': model_name,
                'score': results.get('best_score', 0),
                'n_trials': results.get('n_trials', 0)
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values('score', ascending=False)
        ranking_df['rank'] = range(1, len(ranking_df) + 1)
        
        return ranking_df


def main():
    """Example usage of auto model selection with real data."""
    print("Auto Model Selection example requires real electricity price data.")
    print("Please use the ENTSO-E downloader to get real data first.")
    print("Example usage:")
    print("1. Download real data using ENTSOEDownloader")
    print("2. Preprocess data using DataPreprocessor") 
    print("3. Create AutoModelSelector")
    print("4. Optimize models using optimize_all_models()")
    print("5. Get best model using get_best_model()")
    print("6. Evaluate models using evaluate_models()")


if __name__ == "__main__":
    main()
