"""
Weather data downloader for external features.

This module provides functionality to download weather data from various
sources to use as exogenous features in electricity price forecasting.
"""

import pandas as pd
import requests
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherDownloader:
    """
    Downloader for weather data from various sources.
    
    This class handles the download of weather data including temperature,
    solar radiation, wind speed, and other meteorological variables.
    """
    
    def __init__(self, api_key: Optional[str] = None, source: str = 'openweather'):
        """
        Initialize the weather downloader.
        
        Args:
            api_key: API key for weather service. If None, will try to load from environment.
            source: Weather data source ('openweather', 'meteostat', 'dummy')
        """
        self.api_key = api_key or self._load_api_key()
        self.source = source
        
        if source == 'openweather' and not self.api_key:
            logger.warning("No API key provided for OpenWeather. Using dummy data.")
            self.source = 'dummy'
    
    def _load_api_key(self) -> Optional[str]:
        """Load API key from environment variables."""
        import os
        return os.getenv('OPENWEATHER_API_KEY')
    
    def download_weather_data(
        self,
        city: str,
        country: str,
        start_date: str,
        end_date: str,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Download weather data for a specific location and time period.
        
        Args:
            city: City name
            country: Country code
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            lat: Latitude (optional, will be geocoded if not provided)
            lon: Longitude (optional, will be geocoded if not provided)
            save_path: Path to save the data. If None, returns DataFrame only
            
        Returns:
            DataFrame with weather data
        """
        logger.info(f"Downloading weather data for {city}, {country} from {start_date} to {end_date}")
        
        if self.source == 'openweather':
            df = self._download_openweather(city, country, start_date, end_date, lat, lon)
        elif self.source == 'meteostat':
            df = self._download_meteostat(city, country, start_date, end_date, lat, lon)
        else:
            df = self._generate_dummy_data(start_date, end_date)
        
        if save_path:
            df.to_csv(save_path, index=False)
            logger.info(f"Weather data saved to {save_path}")
        
        return df
    
    def _download_openweather(
        self, 
        city: str, 
        country: str, 
        start_date: str, 
        end_date: str,
        lat: Optional[float] = None,
        lon: Optional[float] = None
    ) -> pd.DataFrame:
        """Download weather data from OpenWeatherMap API."""
        try:
            # Get coordinates if not provided
            if lat is None or lon is None:
                lat, lon = self._geocode_location(city, country)
            
            # OpenWeatherMap Historical API
            base_url = "https://history.openweathermap.org/data/2.5/history/city"
            
            # Download data day by day (API limitation)
            all_data = []
            current_date = datetime.strptime(start_date, '%Y-%m-%d')
            end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
            
            while current_date < end_datetime:
                # OpenWeatherMap provides hourly data for each day
                date_str = current_date.strftime('%Y-%m-%d')
                
                params = {
                    'lat': lat,
                    'lon': lon,
                    'type': 'hour',
                    'start': int(current_date.timestamp()),
                    'end': int((current_date + timedelta(days=1)).timestamp()),
                    'appid': self.api_key
                }
                
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                # Process hourly data
                for hour_data in data.get('list', []):
                    dt = datetime.fromtimestamp(hour_data['dt'])
                    main = hour_data['main']
                    weather = hour_data['weather'][0]
                    wind = hour_data.get('wind', {})
                    clouds = hour_data.get('clouds', {})
                    
                    all_data.append({
                        'datetime': dt,
                        'temperature': main['temp'] - 273.15,  # Convert from Kelvin
                        'humidity': main['humidity'],
                        'pressure': main['pressure'],
                        'wind_speed': wind.get('speed', 0),
                        'wind_direction': wind.get('deg', 0),
                        'cloud_cover': clouds.get('all', 0),
                        'weather_description': weather['description'],
                        'visibility': hour_data.get('visibility', 0) / 1000,  # Convert to km
                        'city': city,
                        'country': country
                    })
                
                current_date += timedelta(days=1)
                
                # Rate limiting
                import time
                time.sleep(0.1)
            
            return pd.DataFrame(all_data)
            
        except Exception as e:
            logger.error(f"Error downloading from OpenWeather: {e}")
            return self._generate_dummy_data(start_date, end_date)
    
    def _download_meteostat(
        self, 
        city: str, 
        country: str, 
        start_date: str, 
        end_date: str,
        lat: Optional[float] = None,
        lon: Optional[float] = None
    ) -> pd.DataFrame:
        """Download weather data from Meteostat API."""
        try:
            # Meteostat API implementation
            # This is a placeholder - you would need to implement the actual API calls
            logger.info("Meteostat API not implemented yet, using dummy data")
            return self._generate_dummy_data(start_date, end_date)
            
        except Exception as e:
            logger.error(f"Error downloading from Meteostat: {e}")
            return self._generate_dummy_data(start_date, end_date)
    
    def _geocode_location(self, city: str, country: str) -> Tuple[float, float]:
        """Get latitude and longitude for a city using OpenWeatherMap geocoding."""
        try:
            geocode_url = "http://api.openweathermap.org/geo/1.0/direct"
            params = {
                'q': f"{city},{country}",
                'limit': 1,
                'appid': self.api_key
            }
            
            response = requests.get(geocode_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if data:
                return data[0]['lat'], data[0]['lon']
            else:
                raise ValueError(f"Could not geocode {city}, {country}")
                
        except Exception as e:
            logger.error(f"Geocoding failed: {e}")
            # Return default coordinates (Rome, Italy)
            return 41.9028, 12.4964
    
    def _generate_dummy_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate dummy weather data for testing purposes."""
        import numpy as np
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Generate hourly timestamps
        timestamps = pd.date_range(start, end, freq='H', inclusive='left')
        
        # Generate realistic weather patterns
        np.random.seed(42)  # For reproducibility
        
        # Temperature with seasonal variation
        days_from_start = (timestamps - start).days
        seasonal_temp = 15 + 10 * np.sin(2 * np.pi * days_from_start / 365.25)
        daily_variation = 5 * np.sin(2 * np.pi * timestamps.hour / 24)
        noise = np.random.normal(0, 2, len(timestamps))
        temperature = seasonal_temp + daily_variation + noise
        
        # Other weather variables
        humidity = np.clip(60 + 20 * np.sin(2 * np.pi * timestamps.hour / 24) + 
                          np.random.normal(0, 10, len(timestamps)), 0, 100)
        
        pressure = 1013 + np.random.normal(0, 10, len(timestamps))
        
        wind_speed = np.clip(np.random.exponential(3, len(timestamps)), 0, 20)
        
        wind_direction = np.random.uniform(0, 360, len(timestamps))
        
        cloud_cover = np.clip(np.random.beta(2, 2, len(timestamps)) * 100, 0, 100)
        
        df = pd.DataFrame({
            'datetime': timestamps,
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure,
            'wind_speed': wind_speed,
            'wind_direction': wind_direction,
            'cloud_cover': cloud_cover,
            'weather_description': 'clear sky',
            'visibility': np.random.uniform(5, 15, len(timestamps)),
            'city': 'Rome',
            'country': 'IT'
        })
        
        logger.info("Generated dummy weather data")
        return df
    
    def get_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract relevant weather features for electricity price forecasting.
        
        Args:
            df: Raw weather DataFrame
            
        Returns:
            DataFrame with engineered weather features
        """
        if df.empty:
            return df
        
        features_df = df.copy()
        
        # Temperature features
        features_df['temp_squared'] = features_df['temperature'] ** 2
        features_df['temp_cooling'] = np.maximum(0, features_df['temperature'] - 20)
        features_df['temp_heating'] = np.maximum(0, 15 - features_df['temperature'])
        
        # Wind features
        features_df['wind_power'] = features_df['wind_speed'] ** 3  # Wind power is proportional to v^3
        
        # Solar features (simplified)
        features_df['solar_potential'] = np.maximum(0, 
            np.sin(np.pi * features_df['datetime'].dt.hour / 12) * 
            (1 - features_df['cloud_cover'] / 100)
        )
        
        # Weather categories
        features_df['is_clear'] = (features_df['cloud_cover'] < 25).astype(int)
        features_df['is_overcast'] = (features_df['cloud_cover'] > 75).astype(int)
        
        # Lag features
        for lag in [1, 24, 168]:  # 1 hour, 1 day, 1 week
            features_df[f'temp_lag_{lag}'] = features_df['temperature'].shift(lag)
            features_df[f'wind_lag_{lag}'] = features_df['wind_speed'].shift(lag)
        
        return features_df


def main():
    """Example usage of the WeatherDownloader."""
    downloader = WeatherDownloader(source='dummy')  # Use dummy data for testing
    
    try:
        df = downloader.download_weather_data(
            city='Rome',
            country='IT',
            start_date='2023-01-01',
            end_date='2023-01-07',
            save_path='data/external/weather_rome_2023.csv'
        )
        print(f"Downloaded {len(df)} weather records")
        print(df.head())
        
        # Extract features
        features_df = downloader.get_weather_features(df)
        print(f"Generated {len(features_df.columns)} weather features")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
