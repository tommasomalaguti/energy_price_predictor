"""
Configuration loader for electricity price forecasting project.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """
    Configuration loader for the electricity price forecasting project.
    
    This class handles loading configuration from YAML files and environment variables.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to configuration file. If None, uses default path.
        """
        if config_path is None:
            # Default config path
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            print(f"Configuration file not found: {self.config_path}")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing configuration file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'data': {
                'country': 'IT',
                'start_date': '2022-01-01',
                'end_date': '2024-01-01',
                'data_type': 'day_ahead'
            },
            'preprocessing': {
                'outlier_threshold': 3.0,
                'missing_threshold': 0.1,
                'price_min': -100,
                'price_max': 1000,
                'interpolation_method': 'linear'
            },
            'models': {
                'baseline': {'enabled': True},
                'ml': {'enabled': True, 'tune_hyperparameters': False},
                'time_series': {'enabled': True}
            },
            'evaluation': {
                'test_size': 0.2,
                'metrics': ['rmse', 'mae', 'mape', 'directional_accuracy', 'r2']
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'data.country')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_api_token(self, service: str) -> Optional[str]:
        """
        Get API token from environment variables.
        
        Args:
            service: Service name ('entsoe' or 'openweather')
            
        Returns:
            API token or None
        """
        token_map = {
            'entsoe': 'ENTSOE_API_TOKEN',
            'openweather': 'OPENWEATHER_API_KEY'
        }
        
        env_var = token_map.get(service.lower())
        if env_var:
            return os.getenv(env_var)
        return None
    
    def update_config(self, key: str, value: Any) -> None:
        """
        Update configuration value.
        
        Args:
            key: Configuration key (e.g., 'data.country')
            value: New value
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def save_config(self, file_path: Optional[str] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            file_path: Path to save configuration. If None, uses original path.
        """
        save_path = file_path or self.config_path
        
        with open(save_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.get('data', {})
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self.get('preprocessing', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.get('models', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.get('evaluation', {})
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration."""
        return self.get('visualization', {})


def main():
    """Example usage of ConfigLoader."""
    # Load configuration
    config = ConfigLoader()
    
    # Get specific values
    country = config.get('data.country')
    test_size = config.get('evaluation.test_size')
    
    print(f"Country: {country}")
    print(f"Test size: {test_size}")
    
    # Get API token
    entsoe_token = config.get_api_token('entsoe')
    print(f"ENTSO-E token available: {entsoe_token is not None}")
    
    # Update configuration
    config.update_config('data.country', 'DE')
    print(f"Updated country: {config.get('data.country')}")


if __name__ == "__main__":
    main()
