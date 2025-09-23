"""
Hyperparameter tuning for electricity price forecasting models.

This module provides automated hyperparameter optimization using Optuna
for various machine learning and time series models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Install with: pip install optuna")

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Class for automated hyperparameter tuning of forecasting models.
    """
    
    def __init__(
        self, 
        n_trials: int = 100, 
        cv_folds: int = 5,
        timeout: Optional[int] = None,
        random_state: int = 42
    ):
        """
        Initialize hyperparameter tuner.
        
        Args:
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
            timeout: Maximum time in seconds for optimization
            random_state: Random state for reproducibility
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter tuning. Install with: pip install optuna")
        
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.timeout = timeout
        self.random_state = random_state
        
        # Initialize Optuna study
        self.study = None
        self.best_params = {}
        self.best_score = None
    
    def optimize_linear_regression(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for linear regression models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary with best parameters
        """
        from sklearn.linear_model import Ridge, Lasso, ElasticNet
        
        def objective(trial):
            # Suggest model type
            model_type = trial.suggest_categorical('model_type', ['ridge', 'lasso', 'elastic_net'])
            
            if model_type == 'ridge':
                alpha = trial.suggest_float('alpha', 1e-6, 1e2, log=True)
                model = Ridge(alpha=alpha, random_state=self.random_state)
            elif model_type == 'lasso':
                alpha = trial.suggest_float('alpha', 1e-6, 1e2, log=True)
                model = Lasso(alpha=alpha, random_state=self.random_state, max_iter=2000)
            else:  # elastic_net
                alpha = trial.suggest_float('alpha', 1e-6, 1e2, log=True)
                l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=self.random_state, max_iter=2000)
            
            # Cross-validation
            cv = TimeSeriesSplit(n_splits=self.cv_folds)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
            
            return scores.mean()
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner()
        )
        
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        return self.best_params
    
    def optimize_random_forest(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for Random Forest.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary with best parameters
        """
        from sklearn.ensemble import RandomForestRegressor
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': self.random_state,
                'n_jobs': -1
            }
            
            model = RandomForestRegressor(**params)
            
            # Cross-validation
            cv = TimeSeriesSplit(n_splits=self.cv_folds)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
            
            return scores.mean()
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner()
        )
        
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        return self.best_params
    
    def optimize_gradient_boosting(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for Gradient Boosting.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary with best parameters
        """
        from sklearn.ensemble import GradientBoostingRegressor
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': self.random_state
            }
            
            model = GradientBoostingRegressor(**params)
            
            # Cross-validation
            cv = TimeSeriesSplit(n_splits=self.cv_folds)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
            
            return scores.mean()
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner()
        )
        
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        return self.best_params
    
    def optimize_xgboost(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for XGBoost.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary with best parameters
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost is required for this optimization")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': self.random_state
            }
            
            model = xgb.XGBRegressor(**params)
            
            # Cross-validation
            cv = TimeSeriesSplit(n_splits=self.cv_folds)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
            
            return scores.mean()
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner()
        )
        
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        return self.best_params
    
    def optimize_lightgbm(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for LightGBM.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary with best parameters
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM is required for this optimization")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': self.random_state,
                'verbose': -1
            }
            
            model = lgb.LGBMRegressor(**params)
            
            # Cross-validation
            cv = TimeSeriesSplit(n_splits=self.cv_folds)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
            
            return scores.mean()
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner()
        )
        
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        return self.best_params
    
    def optimize_svr(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for Support Vector Regression.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary with best parameters
        """
        from sklearn.svm import SVR
        
        def objective(trial):
            params = {
                'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']) or trial.suggest_float('gamma', 1e-4, 1e-1, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
                'epsilon': trial.suggest_float('epsilon', 1e-3, 1e-1, log=True)
            }
            
            model = SVR(**params)
            
            # Cross-validation
            cv = TimeSeriesSplit(n_splits=self.cv_folds)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
            
            return scores.mean()
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner()
        )
        
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        return self.best_params
    
    def optimize_mlp(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for Multi-layer Perceptron.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary with best parameters
        """
        from sklearn.neural_network import MLPRegressor
        
        def objective(trial):
            # Suggest hidden layer sizes
            n_layers = trial.suggest_int('n_layers', 1, 3)
            hidden_layer_sizes = []
            for i in range(n_layers):
                hidden_layer_sizes.append(trial.suggest_int(f'n_neurons_l{i}', 10, 200))
            
            params = {
                'hidden_layer_sizes': tuple(hidden_layer_sizes),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']),
                'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs']),
                'alpha': trial.suggest_float('alpha', 1e-6, 1e-2, log=True),
                'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
                'max_iter': 500,
                'random_state': self.random_state
            }
            
            model = MLPRegressor(**params)
            
            # Cross-validation
            cv = TimeSeriesSplit(n_splits=self.cv_folds)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
            
            return scores.mean()
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner()
        )
        
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        return self.best_params
    
    def optimize_all_models(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        models_to_optimize: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Optimize hyperparameters for multiple models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            models_to_optimize: List of model names to optimize
            
        Returns:
            Dictionary with best parameters for each model
        """
        if models_to_optimize is None:
            models_to_optimize = [
                'linear_regression', 'random_forest', 'gradient_boosting',
                'xgboost', 'lightgbm', 'svr', 'mlp'
            ]
        
        all_best_params = {}
        
        for model_name in models_to_optimize:
            logger.info(f"Optimizing {model_name}...")
            
            try:
                if model_name == 'linear_regression':
                    best_params = self.optimize_linear_regression(X_train, y_train)
                elif model_name == 'random_forest':
                    best_params = self.optimize_random_forest(X_train, y_train)
                elif model_name == 'gradient_boosting':
                    best_params = self.optimize_gradient_boosting(X_train, y_train)
                elif model_name == 'xgboost':
                    best_params = self.optimize_xgboost(X_train, y_train)
                elif model_name == 'lightgbm':
                    best_params = self.optimize_lightgbm(X_train, y_train)
                elif model_name == 'svr':
                    best_params = self.optimize_svr(X_train, y_train)
                elif model_name == 'mlp':
                    best_params = self.optimize_mlp(X_train, y_train)
                else:
                    logger.warning(f"Unknown model: {model_name}")
                    continue
                
                all_best_params[model_name] = best_params
                logger.info(f"Best {model_name} score: {self.best_score:.4f}")
                
            except Exception as e:
                logger.error(f"Error optimizing {model_name}: {e}")
                continue
        
        return all_best_params
    
    def plot_optimization_history(self) -> None:
        """
        Plot optimization history.
        """
        if self.study is None:
            logger.warning("No optimization study available")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot optimization history
            trials = self.study.trials
            values = [t.value for t in trials if t.value is not None]
            trials_num = list(range(len(values)))
            
            ax1.plot(trials_num, values, 'b-', alpha=0.7)
            ax1.set_xlabel('Trial')
            ax1.set_ylabel('Score')
            ax1.set_title('Optimization History')
            ax1.grid(True, alpha=0.3)
            
            # Plot parameter importance
            if len(trials) > 0:
                try:
                    importance = optuna.importance.get_param_importances(self.study)
                    params = list(importance.keys())
                    importances = list(importance.values())
                    
                    ax2.barh(params, importances)
                    ax2.set_xlabel('Importance')
                    ax2.set_title('Parameter Importance')
                    ax2.grid(True, alpha=0.3)
                except Exception as e:
                    logger.warning(f"Could not plot parameter importance: {e}")
                    ax2.text(0.5, 0.5, 'Parameter importance\nnot available', 
                            ha='center', va='center', transform=ax2.transAxes)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")


def main():
    """Example usage of hyperparameter tuning."""
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Create features
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'feature4': np.random.randn(n_samples),
        'feature5': np.random.randn(n_samples)
    })
    
    # Create target with some noise
    y = (2 * X['feature1'] + 3 * X['feature2'] + 
         1.5 * X['feature3'] + 0.5 * X['feature4'] + 
         np.random.randn(n_samples) * 0.5)
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Initialize tuner
    tuner = HyperparameterTuner(n_trials=20, cv_folds=3)
    
    # Optimize Random Forest
    print("Optimizing Random Forest...")
    best_params = tuner.optimize_random_forest(X_train, y_train)
    print(f"Best parameters: {best_params}")
    print(f"Best score: {tuner.best_score:.4f}")
    
    # Plot optimization history
    tuner.plot_optimization_history()


if __name__ == "__main__":
    main()
