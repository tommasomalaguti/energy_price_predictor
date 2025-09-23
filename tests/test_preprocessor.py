"""
Tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    def test_init(self):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor()
        assert preprocessor is not None
        assert preprocessor.price_data is None
        assert preprocessor.weather_data is None
    
    def test_clean_price_data_basic(self, sample_price_data):
        """Test basic price data cleaning."""
        preprocessor = DataPreprocessor()
        preprocessor.price_data = sample_price_data.copy()
        
        cleaned_data = preprocessor.clean_price_data()
        
        assert len(cleaned_data) > 0
        assert 'price' in cleaned_data.columns
        assert cleaned_data['price'].isnull().sum() == 0
        assert not cleaned_data['price'].isna().any()
    
    def test_clean_price_data_outliers(self, sample_price_data):
        """Test outlier removal in price data."""
        preprocessor = DataPreprocessor()
        
        # Add extreme outliers
        data_with_outliers = sample_price_data.copy()
        data_with_outliers.loc[data_with_outliers.index[0], 'price'] = 10000  # Extreme outlier
        data_with_outliers.loc[data_with_outliers.index[1], 'price'] = -5000  # Extreme outlier
        
        preprocessor.price_data = data_with_outliers
        cleaned_data = preprocessor.clean_price_data()
        
        # Check that extreme outliers are removed
        assert cleaned_data['price'].max() < 1000
        assert cleaned_data['price'].min() > -1000
    
    def test_engineer_features_basic(self, sample_price_data):
        """Test basic feature engineering."""
        preprocessor = DataPreprocessor()
        preprocessor.price_data = sample_price_data.copy()
        
        features = preprocessor.engineer_features()
        
        # Check that features are created
        assert len(features) > 0
        assert 'hour' in features.columns
        assert 'day_of_week' in features.columns
        assert 'month' in features.columns
        assert 'year' in features.columns
        
        # Check lag features
        lag_cols = [col for col in features.columns if col.startswith('price_lag_')]
        assert len(lag_cols) > 0
        
        # Check rolling features
        rolling_cols = [col for col in features.columns if col.startswith('price_rolling_')]
        assert len(rolling_cols) > 0
    
    def test_engineer_features_with_weather(self, sample_price_data):
        """Test feature engineering with weather data."""
        preprocessor = DataPreprocessor()
        preprocessor.price_data = sample_price_data.copy()
        
        # Create mock weather data
        weather_data = pd.DataFrame({
            'datetime': sample_price_data.index,
            'temperature': np.random.normal(15, 5, len(sample_price_data)),
            'humidity': np.random.uniform(30, 90, len(sample_price_data)),
            'wind_speed': np.random.exponential(5, len(sample_price_data))
        })
        weather_data.set_index('datetime', inplace=True)
        preprocessor.weather_data = weather_data
        
        features = preprocessor.engineer_features()
        
        # Check that weather features are included
        assert 'temperature' in features.columns
        assert 'humidity' in features.columns
        assert 'wind_speed' in features.columns
    
    def test_handle_missing_values(self, sample_price_data):
        """Test missing value handling."""
        preprocessor = DataPreprocessor()
        
        # Create data with missing values
        data_with_nans = sample_price_data.copy()
        data_with_nans.loc[data_with_nans.index[10:20], 'price'] = np.nan
        
        preprocessor.price_data = data_with_nans
        cleaned_data = preprocessor.clean_price_data()
        
        # Check that missing values are handled
        assert cleaned_data['price'].isnull().sum() == 0
    
    def test_price_range_validation(self, sample_price_data):
        """Test price range validation."""
        preprocessor = DataPreprocessor()
        
        # Create data with prices outside reasonable range
        data_extreme = sample_price_data.copy()
        data_extreme.loc[data_extreme.index[0], 'price'] = 2000  # Too high
        data_extreme.loc[data_extreme.index[1], 'price'] = -500  # Too low
        
        preprocessor.price_data = data_extreme
        cleaned_data = preprocessor.clean_price_data()
        
        # Check that extreme values are handled
        assert cleaned_data['price'].max() < 1000
        assert cleaned_data['price'].min() > -200
    
    def test_feature_engineering_empty_data(self):
        """Test feature engineering with empty data."""
        preprocessor = DataPreprocessor()
        preprocessor.price_data = pd.DataFrame()
        
        with pytest.raises(ValueError):
            preprocessor.engineer_features()
    
    def test_feature_engineering_missing_datetime_index(self):
        """Test feature engineering with data missing datetime index."""
        preprocessor = DataPreprocessor()
        
        # Create data without datetime index
        data_no_index = pd.DataFrame({
            'price': [1, 2, 3, 4, 5]
        })
        preprocessor.price_data = data_no_index
        
        with pytest.raises(ValueError):
            preprocessor.engineer_features()
