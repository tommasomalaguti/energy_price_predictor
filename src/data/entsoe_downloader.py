"""
ENTSO-E data downloader for electricity market prices.

This module provides functionality to download day-ahead and intraday
electricity market price data from the ENTSO-E Transparency Platform.
"""

import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ENTSOEDownloader:
    """
    Downloader for ENTSO-E electricity market data.
    
    This class handles the download of day-ahead and intraday electricity
    market prices from the ENTSO-E Transparency Platform API.
    """
    
    # ENTSO-E API endpoints
    BASE_URL = "https://web-api.tp.entsoe.eu/api"
    
    # Country codes mapping
    COUNTRY_CODES = {
        'AT': 'Austria',
        'BE': 'Belgium', 
        'BG': 'Bulgaria',
        'HR': 'Croatia',
        'CY': 'Cyprus',
        'CZ': 'Czech Republic',
        'DK': 'Denmark',
        'EE': 'Estonia',
        'FI': 'Finland',
        'FR': 'France',
        'DE': 'Germany',
        'GR': 'Greece',
        'HU': 'Hungary',
        'IE': 'Ireland',
        'IT': 'Italy',
        'LV': 'Latvia',
        'LT': 'Lithuania',
        'LU': 'Luxembourg',
        'MT': 'Malta',
        'NL': 'Netherlands',
        'PL': 'Poland',
        'PT': 'Portugal',
        'RO': 'Romania',
        'SK': 'Slovakia',
        'SI': 'Slovenia',
        'ES': 'Spain',
        'SE': 'Sweden'
    }
    
    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize the ENTSO-E downloader.
        
        Args:
            api_token: ENTSO-E API token. If None, will try to load from environment.
        """
        self.api_token = api_token or self._load_api_token()
        if not self.api_token:
            raise ValueError("ENTSO-E API token is required. Please provide it or set ENTSOE_API_TOKEN environment variable.")
        
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/xml'
        })
    
    def _load_api_token(self) -> Optional[str]:
        """Load API token from environment variables."""
        import os
        return os.getenv('ENTSOE_API_TOKEN')
    
    def download_price_data(
        self,
        country: str,
        start_date: str,
        end_date: Optional[str] = None,
        data_type: str = 'day_ahead',
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Download electricity price data for a specific country and time period.
        
        Args:
            country: Country code (e.g., 'IT', 'DE', 'FR')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format. If None, downloads until today
            data_type: Type of data ('day_ahead' or 'intraday')
            save_path: Path to save the data. If None, returns DataFrame only
            
        Returns:
            DataFrame with price data
        """
        if country not in self.COUNTRY_CODES:
            raise ValueError(f"Invalid country code. Available: {list(self.COUNTRY_CODES.keys())}")
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Downloading {data_type} price data for {country} from {start_date} to {end_date}")
        
        # Download data in chunks to avoid API limits
        all_data = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current_date < end_datetime:
            chunk_end = min(current_date + timedelta(days=30), end_datetime)
            
            try:
                chunk_data = self._download_chunk(
                    country, 
                    current_date.strftime('%Y-%m-%d'),
                    chunk_end.strftime('%Y-%m-%d'),
                    data_type
                )
                if not chunk_data.empty:
                    all_data.append(chunk_data)
                
                # Rate limiting
                time.sleep(1)
                current_date = chunk_end
                
            except Exception as e:
                logger.error(f"Error downloading chunk {current_date}: {e}")
                current_date = chunk_end
                continue
        
        if not all_data:
            logger.warning("No data downloaded")
            return pd.DataFrame()
        
        # Combine all chunks
        df = pd.concat(all_data, ignore_index=True)
        df = self._process_price_data(df)
        
        if save_path:
            df.to_csv(save_path, index=False)
            logger.info(f"Data saved to {save_path}")
        
        return df
    
    def _download_chunk(
        self, 
        country: str, 
        start_date: str, 
        end_date: str, 
        data_type: str
    ) -> pd.DataFrame:
        """Download a chunk of data to avoid API limits."""
        
        # Construct the API request
        if data_type == 'day_ahead':
            document_type = 'A44'  # Day-ahead prices
        else:
            document_type = 'A65'  # Intraday prices
        
        params = {
            'documentType': document_type,
            'in_Domain': f'10Y{country}----------',
            'out_Domain': f'10Y{country}----------',
            'periodStart': f'{start_date}T00:00Z',
            'periodEnd': f'{end_date}T23:59Z',
            'securityToken': self.api_token
        }
        
        try:
            logger.info(f"Making API request with params: {params}")
            response = self.session.get(f"{self.BASE_URL}", params=params)
            
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")
            
            if response.status_code == 401:
                logger.error("401 Unauthorized - Check your API token")
                logger.error(f"Response text: {response.text[:500]}")
                return pd.DataFrame()
            
            response.raise_for_status()
            
            # Parse XML response
            df = self._parse_xml_response(response.text, data_type)
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response text: {e.response.text[:500]}")
            return pd.DataFrame()
    
    def _parse_xml_response(self, xml_content: str, data_type: str) -> pd.DataFrame:
        """Parse XML response from ENTSO-E API."""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(xml_content, 'xml')
            
            # Find all time series
            time_series = soup.find_all('TimeSeries')
            
            data = []
            for ts in time_series:
                # Extract price data
                points = ts.find_all('Point')
                for point in points:
                    position = int(point.find('position').text)
                    price = float(point.find('price.amount').text)
                    
                    # Calculate actual datetime
                    start_time = ts.find('start').text
                    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    actual_dt = start_dt + timedelta(hours=position-1)
                    
                    data.append({
                        'datetime': actual_dt,
                        'price': price,
                        'type': data_type
                    })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error parsing XML: {e}")
            return pd.DataFrame()
    
    def _process_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean the downloaded price data."""
        if df.empty:
            return df
        
        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['datetime']).reset_index(drop=True)
        
        # Handle missing values
        df['price'] = df['price'].fillna(method='ffill')
        
        # Add time-based features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['year'] = df['datetime'].dt.year
        
        return df
    
    def get_available_countries(self) -> Dict[str, str]:
        """Get list of available countries."""
        return self.COUNTRY_CODES.copy()
    
    def validate_date_range(self, start_date: str, end_date: str) -> bool:
        """Validate that the date range is reasonable."""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        if start >= end:
            return False
        
        if (end - start).days > 365:
            logger.warning("Date range is longer than 1 year. Consider downloading in smaller chunks.")
        
        return True
    
    def test_api_connection(self) -> bool:
        """Test if the API token is valid by making a simple request."""
        try:
            # Try to get a small amount of data
            test_params = {
                'documentType': 'A44',
                'in_Domain': '10YIT----------',
                'out_Domain': '10YIT----------',
                'periodStart': '2024-01-01T00:00Z',
                'periodEnd': '2024-01-01T23:59Z',
                'securityToken': self.api_token
            }
            
            response = self.session.get(f"{self.BASE_URL}", params=test_params)
            
            if response.status_code == 200:
                logger.info("✓ API token is valid")
                return True
            elif response.status_code == 401:
                logger.error("✗ API token is invalid or not activated")
                return False
            else:
                logger.warning(f"Unexpected response: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"API test failed: {e}")
            return False


def main():
    """Example usage of the ENTSOEDownloader."""
    # Example: Download Italian day-ahead prices for the last 2 years
    downloader = ENTSOEDownloader()
    
    # You need to set your API token
    # downloader = ENTSOEDownloader(api_token="your_token_here")
    
    try:
        df = downloader.download_price_data(
            country='IT',
            start_date='2022-01-01',
            end_date='2024-01-01',
            data_type='day_ahead',
            save_path='data/raw/italy_prices_2022_2024.csv'
        )
        print(f"Downloaded {len(df)} records")
        print(df.head())
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please set your ENTSO-E API token in the ENTSOE_API_TOKEN environment variable")


if __name__ == "__main__":
    main()
