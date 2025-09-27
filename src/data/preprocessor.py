"""
Data preprocessing utilities for electricity price forecasting.

This module provides comprehensive data cleaning, validation, and preprocessing
functionality for electricity market data and external features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Union
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing for electricity price forecasting.
    
    This class handles data cleaning, validation, feature engineering,
    and preparation for machine learning models.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config or self._get_default_config()
        self.price_data = None
        self.weather_data = None
        self.processed_data = None
        
    def _get_default_config(self) -> Dict:
        """Get default preprocessing configuration."""
        return {
            'outlier_threshold': 3.0,  # Z-score threshold for outliers
            'missing_threshold': 0.1,  # Maximum fraction of missing values allowed
            'price_min': -100,  # Minimum reasonable price (€/MWh)
            'price_max': 1000,  # Maximum reasonable price (€/MWh)
            'interpolation_method': 'linear',
            'holiday_countries': ['IT', 'DE', 'FR', 'ES'],  # Countries for holiday data
            'timezone': 'Europe/Rome'
        }
    
    def load_price_data(self, file_path: str) -> pd.DataFrame:
        """
        Load electricity price data from CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with price data
        """
        try:
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
            self.price_data = df
            logger.info(f"Loaded price data: {len(df)} records from {df.index.min()} to {df.index.max()}")
            return df
        except Exception as e:
            logger.error(f"Error loading price data: {e}")
            raise
    
    def load_weather_data(self, file_path: str) -> pd.DataFrame:
        """
        Load weather data from CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with weather data
        """
        try:
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
            self.weather_data = df
            logger.info(f"Loaded weather data: {len(df)} records from {df.index.min()} to {df.index.max()}")
            return df
        except Exception as e:
            logger.error(f"Error loading weather data: {e}")
            raise
    
    def clean_price_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clean electricity price data.
        
        Args:
            df: Price DataFrame. If None, uses self.price_data
            
        Returns:
            Cleaned DataFrame
        """
        if df is None:
            df = self.price_data.copy()
        
        if df is None:
            raise ValueError("No price data available. Please load data first.")
        
        logger.info("Cleaning price data...")
        
        # Remove duplicates
        initial_count = len(df)
        df = df[~df.index.duplicated(keep='first')]
        logger.info(f"Removed {initial_count - len(df)} duplicate records")
        
        # Handle missing values first
        df = self._handle_missing_values(df, 'price')
        
        # Handle outliers
        df = self._remove_outliers(df, 'price')
        
        # Validate price range
        df = self._validate_price_range(df)
        
        # Ensure regular time index
        df = self._ensure_regular_time_index(df)
        
        # Check minimum data requirement
        if len(df) < 10:
            logger.warning(f"Very few records remaining ({len(df)}). Consider using sample data or checking data quality.")
        
        self.price_data = df
        logger.info(f"Price data cleaned: {len(df)} records remaining")
        return df
    
    def clean_weather_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clean weather data.
        
        Args:
            df: Weather DataFrame. If None, uses self.weather_data
            
        Returns:
            Cleaned DataFrame
        """
        if df is None:
            df = self.weather_data.copy()
        
        if df is None:
            raise ValueError("No weather data available. Please load data first.")
        
        logger.info("Cleaning weather data...")
        
        # Remove duplicates
        initial_count = len(df)
        df = df[~df.index.duplicated(keep='first')]
        logger.info(f"Removed {initial_count - len(df)} duplicate records")
        
        # Handle missing values for each weather variable
        weather_columns = ['temperature', 'humidity', 'pressure', 'wind_speed', 'cloud_cover']
        for col in weather_columns:
            if col in df.columns:
                df = self._handle_missing_values(df, col)
        
        # Ensure regular time index
        df = self._ensure_regular_time_index(df)
        
        self.weather_data = df
        logger.info(f"Weather data cleaned: {len(df)} records remaining")
        return df
    
    def _remove_outliers(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Remove outliers using Z-score method with more lenient threshold."""
        if column not in df.columns:
            return df
        
        # Use more lenient outlier detection for small datasets
        if len(df) < 100:
            threshold = 5.0  # More lenient for small datasets
        else:
            threshold = self.config['outlier_threshold']
        
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers = z_scores > threshold
        
        if outliers.sum() > 0:
            logger.info(f"Removing {outliers.sum()} outliers from {column} (threshold: {threshold})")
            df = df[~outliers]
        
        return df
    
    def _validate_price_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that prices are within reasonable range."""
        if 'price' not in df.columns:
            return df
        
        valid_prices = (df['price'] >= self.config['price_min']) & (df['price'] <= self.config['price_max'])
        invalid_count = (~valid_prices).sum()
        
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} prices outside reasonable range")
            df = df[valid_prices]
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Handle missing values in a specific column."""
        if column not in df.columns:
            return df
        
        missing_count = df[column].isna().sum()
        missing_fraction = missing_count / len(df)
        
        if missing_fraction > self.config['missing_threshold']:
            logger.warning(f"Column {column} has {missing_fraction:.1%} missing values")
        
        if missing_count > 0:
            if self.config['interpolation_method'] == 'linear':
                df[column] = df[column].interpolate(method='linear')
            elif self.config['interpolation_method'] == 'forward':
                df[column] = df[column].ffill()
            elif self.config['interpolation_method'] == 'backward':
                df[column] = df[column].bfill()
            else:
                df[column] = df[column].fillna(df[column].mean())
        
        return df
    
    def _ensure_regular_time_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure regular hourly time index."""
        if df.empty:
            return df
        
        # Create regular hourly index
        start_time = pd.Timestamp(df.index.min()).floor('h')
        end_time = pd.Timestamp(df.index.max()).ceil('h')
        regular_index = pd.date_range(start_time, end_time, freq='h', inclusive='left')
        
        # Reindex and interpolate missing values
        df = df.reindex(regular_index)
        
        return df
    
    def engineer_features(self, price_df: Optional[pd.DataFrame] = None, 
                         weather_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Engineer features for electricity price forecasting.
        
        Args:
            price_df: Price DataFrame. If None, uses self.price_data
            weather_df: Weather DataFrame. If None, uses self.weather_data
            
        Returns:
            DataFrame with engineered features
        """
        if price_df is None:
            price_df = self.price_data.copy()
        if weather_df is None:
            weather_df = self.weather_data.copy() if self.weather_data is not None else None
        
        if price_df is None:
            raise ValueError("No price data available for feature engineering")
        
        logger.info("Engineering features...")
        
        # Start with price data
        features_df = price_df.copy()
        
        # Time-based features
        if not isinstance(features_df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex for feature engineering")
        features_df = self._add_time_features(features_df)
        
        # Price-based features
        features_df = self._add_price_features(features_df)
        
        # Weather features
        if weather_df is not None and not weather_df.empty:
            features_df = self._add_weather_features(features_df, weather_df)
        
        # Holiday features
        features_df = self._add_holiday_features(features_df)
        
        # Lag features
        features_df = self._add_lag_features(features_df)
        
        # Rolling statistics
        features_df = self._add_rolling_features(features_df)
        
        # Fourier features for seasonality
        features_df = self._add_fourier_features(features_df)
        
        # Advanced time features
        features_df = self._add_advanced_time_features(features_df)
        
        self.processed_data = features_df
        logger.info(f"Feature engineering complete: {len(features_df.columns)} features")
        return features_df
    
    def create_features(self, price_df: Optional[pd.DataFrame] = None, 
                        weather_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Alias for engineer_features method for backward compatibility.
        
        Args:
            price_df: Price DataFrame. If None, uses self.price_data
            weather_df: Weather DataFrame. If None, uses self.weather_data
            
        Returns:
            DataFrame with engineered features
        """
        return self.engineer_features(price_df, weather_df)
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Business hours
        df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        if 'price' not in df.columns:
            return df
        
        # Price statistics
        df['price_mean_24h'] = df['price'].rolling(window=24, min_periods=1).mean()
        df['price_std_24h'] = df['price'].rolling(window=24, min_periods=1).std()
        df['price_min_24h'] = df['price'].rolling(window=24, min_periods=1).min()
        df['price_max_24h'] = df['price'].rolling(window=24, min_periods=1).max()
        
        # Price percentiles
        df['price_q25_24h'] = df['price'].rolling(window=24, min_periods=1).quantile(0.25)
        df['price_q75_24h'] = df['price'].rolling(window=24, min_periods=1).quantile(0.75)
        
        # Price volatility
        df['price_volatility'] = df['price_std_24h'] / df['price_mean_24h']
        
        # Price change
        df['price_change'] = df['price'].diff()
        df['price_change_pct'] = df['price'].pct_change()
        
        return df
    
    def _add_weather_features(self, df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Add weather-based features."""
        # Merge weather data
        df = df.join(weather_df, how='left')
        
        # Temperature features
        if 'temperature' in df.columns:
            df['temp_squared'] = df['temperature'] ** 2
            df['temp_cooling'] = np.maximum(0, df['temperature'] - 20)
            df['temp_heating'] = np.maximum(0, 15 - df['temperature'])
        
        # Wind features
        if 'wind_speed' in df.columns:
            df['wind_power'] = df['wind_speed'] ** 3
        
        # Solar features
        if 'cloud_cover' in df.columns:
            df['solar_potential'] = np.maximum(0, 
                np.sin(np.pi * df['hour'] / 12) * (1 - df['cloud_cover'] / 100)
            )
        
        return df
    
    def _add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add holiday features."""
        # Simple holiday detection (can be enhanced with external holiday data)
        df['is_holiday'] = 0
        
        # New Year
        df.loc[(df.index.month == 1) & (df.index.day == 1), 'is_holiday'] = 1
        
        # Christmas
        df.loc[(df.index.month == 12) & (df.index.day == 25), 'is_holiday'] = 1
        
        # Easter (simplified - first Sunday in April)
        easter_sunday = df.index[(df.index.month == 4) & (df.index.dayofweek == 6) & (df.index.day <= 7)]
        df.loc[easter_sunday, 'is_holiday'] = 1
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features with interactions."""
        if 'price' not in df.columns:
            return df
        
        # Price lags
        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:  # 1h, 2h, 3h, 6h, 12h, 1d, 2d, 1w
            df[f'price_lag_{lag}'] = df['price'].shift(lag)
        
        # Temperature lags (if available)
        if 'temperature' in df.columns:
            for lag in [1, 24, 168]:
                df[f'temp_lag_{lag}'] = df['temperature'].shift(lag)
        
        # Lag interactions (price * hour, price * day_of_week)
        for lag in [1, 24]:
            if f'price_lag_{lag}' in df.columns:
                df[f'price_hour_interaction_lag_{lag}'] = df[f'price_lag_{lag}'] * df['hour']
                df[f'price_day_interaction_lag_{lag}'] = df[f'price_lag_{lag}'] * df['day_of_week']
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced rolling statistical features."""
        if 'price' not in df.columns:
            return df
        
        # Rolling means and standard deviations
        for window in [3, 6, 12, 24, 48, 168]:
            df[f'price_mean_{window}h'] = df['price'].rolling(window=window, min_periods=1).mean()
            df[f'price_std_{window}h'] = df['price'].rolling(window=window, min_periods=1).std()
            df[f'price_median_{window}h'] = df['price'].rolling(window=window, min_periods=1).median()
            df[f'price_skew_{window}h'] = df['price'].rolling(window=window, min_periods=1).skew()
            df[f'price_kurt_{window}h'] = df['price'].rolling(window=window, min_periods=1).kurt()
        
        # Rolling quantiles
        for window in [24, 168]:
            for q in [0.1, 0.25, 0.75, 0.9]:
                df[f'price_q{int(q*100)}_{window}h'] = df['price'].rolling(window=window, min_periods=1).quantile(q)
        
        # Rolling correlations with time features
        df['price_hour_corr'] = df['price'].rolling(window=168, min_periods=24).corr(df['hour'])
        df['price_day_corr'] = df['price'].rolling(window=168, min_periods=24).corr(df['day_of_week'])
        
        # Rolling autocorrelation
        for lag in [1, 24, 168]:
            df[f'price_autocorr_lag_{lag}'] = df['price'].rolling(window=168, min_periods=24).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
            )
        
        # Rolling volatility (GARCH-like)
        df['price_volatility_24h'] = df['price'].rolling(window=24, min_periods=1).std()
        df['price_volatility_168h'] = df['price'].rolling(window=168, min_periods=1).std()
        
        # Rolling momentum and mean reversion
        df['price_momentum_24h'] = df['price'] - df['price'].shift(24)
        df['price_mean_reversion_24h'] = df['price_mean_24h'] - df['price']
        
        return df
    
    def _add_fourier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Fourier terms for capturing seasonal patterns."""
        # Daily seasonality (24-hour cycle)
        for k in range(1, 4):  # First 3 harmonics
            df[f'fourier_daily_sin_{k}'] = np.sin(2 * np.pi * k * df['hour'] / 24)
            df[f'fourier_daily_cos_{k}'] = np.cos(2 * np.pi * k * df['hour'] / 24)
        
        # Weekly seasonality (168-hour cycle)
        hour_of_week = df['hour'] + 24 * df['day_of_week']
        for k in range(1, 3):  # First 2 harmonics
            df[f'fourier_weekly_sin_{k}'] = np.sin(2 * np.pi * k * hour_of_week / 168)
            df[f'fourier_weekly_cos_{k}'] = np.cos(2 * np.pi * k * hour_of_week / 168)
        
        # Annual seasonality (8760-hour cycle)
        hour_of_year = df.index.dayofyear * 24 + df['hour']
        for k in range(1, 2):  # First harmonic only
            df[f'fourier_annual_sin_{k}'] = np.sin(2 * np.pi * k * hour_of_year / 8760)
            df[f'fourier_annual_cos_{k}'] = np.cos(2 * np.pi * k * hour_of_year / 8760)
        
        return df
    
    def _add_advanced_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced time-based features."""
        # Time since last peak/valley
        price_rolling_max = df['price'].rolling(window=24, min_periods=1).max()
        price_rolling_min = df['price'].rolling(window=24, min_periods=1).min()
        
        df['time_since_peak'] = 0
        df['time_since_valley'] = 0
        
        for i in range(1, len(df)):
            if df['price'].iloc[i] == price_rolling_max.iloc[i]:
                df['time_since_peak'].iloc[i] = 0
            else:
                df['time_since_peak'].iloc[i] = df['time_since_peak'].iloc[i-1] + 1
                
            if df['price'].iloc[i] == price_rolling_min.iloc[i]:
                df['time_since_valley'].iloc[i] = 0
            else:
                df['time_since_valley'].iloc[i] = df['time_since_valley'].iloc[i-1] + 1
        
        # Price position within daily range
        daily_max = df['price'].groupby(df.index.date).transform('max')
        daily_min = df['price'].groupby(df.index.date).transform('min')
        df['price_position_daily'] = (df['price'] - daily_min) / (daily_max - daily_min + 1e-8)
        
        # Price position within weekly range
        weekly_max = df['price'].groupby(df.index.isocalendar().week).transform('max')
        weekly_min = df['price'].groupby(df.index.isocalendar().week).transform('min')
        df['price_position_weekly'] = (df['price'] - weekly_min) / (weekly_max - weekly_min + 1e-8)
        
        return df
    
    def prepare_training_data(self, target_column: str = 'price', 
                            test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training machine learning models.
        
        Args:
            target_column: Name of the target column
            test_size: Fraction of data to use for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Run engineer_features() first.")
        
        # Remove rows with missing target values
        df = self.processed_data.dropna(subset=[target_column])
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col != target_column]
        X = df[feature_columns]
        y = df[target_column]
        
        # Split into train and test sets
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Training data: {len(X_train)} samples, {len(X_train.columns)} features")
        logger.info(f"Test data: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, file_path: str) -> None:
        """Save processed data to CSV file."""
        if self.processed_data is None:
            raise ValueError("No processed data available to save.")
        
        self.processed_data.to_csv(file_path)
        logger.info(f"Processed data saved to {file_path}")


def main():
    """Example usage of the DataPreprocessor."""
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Load and clean data
    try:
        # This would be used with actual data files
        # preprocessor.load_price_data('data/raw/italy_prices.csv')
        # preprocessor.load_weather_data('data/external/weather_rome.csv')
        
        # Clean data
        # price_clean = preprocessor.clean_price_data()
        # weather_clean = preprocessor.clean_weather_data()
        
        # Engineer features
        # features = preprocessor.engineer_features()
        
        # Prepare training data
        # X_train, X_test, y_train, y_test = preprocessor.prepare_training_data()
        
        print("DataPreprocessor example completed")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
