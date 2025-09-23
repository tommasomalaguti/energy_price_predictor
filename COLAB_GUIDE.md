# Google Colab Setup Guide

## Quick Start

### 1. Open Google Colab
- Go to https://colab.research.google.com/
- Create a new notebook

### 2. Setup Cell (Run First)
```python
# Install packages and clone repository
!pip install xgboost lightgbm prophet tensorflow torch plotly streamlit
!git clone https://github.com/tommasomalaguti/energy_price_predictor.git
%cd energy_price_predictor

print("Setup complete!")
```

### 3. Import Libraries
```python
import sys
import os

# Add the src directory to Python path
sys.path.append('src')
sys.path.append('.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Check if we can find our modules
print("Checking module paths...")
print("Current directory:", os.getcwd())
print("Files in current directory:", os.listdir('.'))
if os.path.exists('src'):
    print("Files in src directory:", os.listdir('src'))
    if os.path.exists('src/data'):
        print("Files in src/data directory:", os.listdir('src/data'))
    if os.path.exists('src/models'):
        print("Files in src/models directory:", os.listdir('src/models'))

# Import our modules with proper error handling
try:
    from src.data.entsoe_downloader import ENTSOEDownloader
    from src.data.preprocessor import DataPreprocessor
    from src.models.baseline_models import BaselineModels
    from src.models.ml_models import MLModels
    from src.models.time_series_models import TimeSeriesModels
    from src.evaluation.metrics import EvaluationMetrics
    from src.evaluation.visualization import ModelVisualization
    print("✓ All modules imported successfully!")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Trying alternative import paths...")
    
    # Try alternative import paths
    try:
        from data.entsoe_downloader import ENTSOEDownloader
        from data.preprocessor import DataPreprocessor
        from models.baseline_models import BaselineModels
        from models.ml_models import MLModels
        from models.time_series_models import TimeSeriesModels
        from evaluation.metrics import EvaluationMetrics
        from evaluation.visualization import ModelVisualization
        print("✓ Modules imported with alternative paths!")
    except ImportError as e2:
        print(f"✗ Alternative import also failed: {e2}")
        print("Please check the file structure and try again.")
```

### 3.1 Quick Fix for Import Issues
```python
# If you're getting import errors, run this cell first
import os
import sys

# Check current directory structure
print("Current working directory:", os.getcwd())
print("\nDirectory contents:")
for item in os.listdir('.'):
    print(f"  {item}")

# Check if src directory exists and what's in it
if os.path.exists('src'):
    print("\nsrc directory contents:")
    for item in os.listdir('src'):
        print(f"  {item}")
        
    # Check subdirectories
    for subdir in ['data', 'models', 'evaluation']:
        if os.path.exists(f'src/{subdir}'):
            print(f"\nsrc/{subdir} directory contents:")
            for item in os.listdir(f'src/{subdir}'):
                print(f"  {item}")
else:
    print("\n❌ src directory not found!")
    print("Make sure you're in the energy_price_predictor directory")
    print("Run: %cd energy_price_predictor")

# Add all necessary paths
sys.path.append('.')
sys.path.append('src')
sys.path.append('src/data')
sys.path.append('src/models')
sys.path.append('src/evaluation')

print(f"\nPython path updated. Current sys.path:")
for path in sys.path[-5:]:  # Show last 5 paths
    print(f"  {path}")
```

### 3.1.1 Quick Fix for API Token Issue
```python
# Quick fix: Update the ENTSOEDownloader to use correct API authentication
import requests

# Patch the downloader to use securityToken in params instead of headers
def patch_downloader():
    """Apply a quick patch to fix the API authentication."""
    import src.data.entsoe_downloader as entsoe_module
    
    # Store the original _download_chunk method
    original_download_chunk = entsoe_module.ENTSOEDownloader._download_chunk
    
    def patched_download_chunk(self, country, start_date, end_date, data_type):
        """Patched version that uses securityToken in params."""
        if data_type == 'day_ahead':
            document_type = 'A44'
        else:
            document_type = 'A65'
        
        params = {
            'documentType': document_type,
            'in_Domain': f'10Y{country}----------',
            'out_Domain': f'10Y{country}----------',
            'periodStart': f'{start_date.replace("-", "")}0000',
            'periodEnd': f'{end_date.replace("-", "")}2359',
            'securityToken': self.api_token  # This is the key fix!
        }
        
        try:
            response = self.session.get(f"{self.BASE_URL}", params=params)
            response.raise_for_status()
            
            # Parse XML response
            df = self._parse_xml_response(response.text, data_type)
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response text: {e.response.text[:200]}")
            return pd.DataFrame()
    
    # Apply the patch
    entsoe_module.ENTSOEDownloader._download_chunk = patched_download_chunk
    print("✓ ENTSOEDownloader patched successfully!")

# Apply the patch
patch_downloader()
```

### 3.2 Real Data Test - Updated Approach Based on Research
```python
# Let's try real data with corrected parameters based on ENTSO-E API research
import requests
from datetime import datetime, timedelta

ENTSOE_API_TOKEN = "2c8cd8e0-0a84-4f67-90ba-b79d07ab2667"

print("🔍 Trying real data with corrected parameters...")

# Test 1: Try Germany with correct domain code and recent date
print("\n1. Testing Germany with correct domain code...")
today = datetime.now()
yesterday = today - timedelta(days=1)
yesterday_str = yesterday.strftime('%Y%m%d')

# Use the correct German domain code from research
de_corrected_params = {
    'documentType': 'A44',
    'in_Domain': '10Y1001A1001A63L',  # Correct German domain code
    'out_Domain': '10Y1001A1001A63L',
    'periodStart': f'{yesterday_str}0000',
    'periodEnd': f'{yesterday_str}2359',
    'securityToken': ENTSOE_API_TOKEN
}

response = requests.get("https://web-api.tp.entsoe.eu/api", params=de_corrected_params)
print(f"Germany corrected: {response.status_code}")
if response.status_code == 200:
    print("✅ SUCCESS! Real data from Germany!")
    print("Response preview:", response.text[:300])
else:
    print(f"Response: {response.text[:300]}")

# Test 2: Try with even smaller time window (just 6 hours)
if response.status_code != 200:
    print("\n2. Trying smaller time window (6 hours)...")
    small_window_params = {
        'documentType': 'A44',
        'in_Domain': '10Y1001A1001A63L',
        'out_Domain': '10Y1001A1001A63L',
        'periodStart': f'{yesterday_str}0000',
        'periodEnd': f'{yesterday_str}0600',  # Only 6 hours
        'securityToken': ENTSOE_API_TOKEN
    }
    
    response = requests.get("https://web-api.tp.entsoe.eu/api", params=small_window_params)
    print(f"Small window: {response.status_code}")
    if response.status_code == 200:
        print("✅ SUCCESS with smaller window!")
        print("Response preview:", response.text[:300])
    else:
        print(f"Response: {response.text[:300]}")

# Test 3: Try France with correct domain code
if response.status_code != 200:
    print("\n3. Trying France with correct domain code...")
    fr_params = {
        'documentType': 'A44',
        'in_Domain': '10YFR-RTE------C',  # French domain code
        'out_Domain': '10YFR-RTE------C',
        'periodStart': f'{yesterday_str}0000',
        'periodEnd': f'{yesterday_str}0600',
        'securityToken': ENTSOE_API_TOKEN
    }
    
    response = requests.get("https://web-api.tp.entsoe.eu/api", params=fr_params)
    print(f"France: {response.status_code}")
    if response.status_code == 200:
        print("✅ SUCCESS with France!")
        print("Response preview:", response.text[:300])
    else:
        print(f"Response: {response.text[:300]}")

# Test 4: Try Netherlands
if response.status_code != 200:
    print("\n4. Trying Netherlands...")
    nl_params = {
        'documentType': 'A44',
        'in_Domain': '10YNL----------L',  # Dutch domain code
        'out_Domain': '10YNL----------L',
        'periodStart': f'{yesterday_str}0000',
        'periodEnd': f'{yesterday_str}0600',
        'securityToken': ENTSOE_API_TOKEN
    }
    
    response = requests.get("https://web-api.tp.entsoe.eu/api", params=nl_params)
    print(f"Netherlands: {response.status_code}")
    if response.status_code == 200:
        print("✅ SUCCESS with Netherlands!")
        print("Response preview:", response.text[:300])
    else:
        print(f"Response: {response.text[:300]}")

# If we got real data, let's try to parse it
if response.status_code == 200:
    print("\n🎉 REAL DATA SUCCESS! Let's parse it...")
    try:
        from bs4 import BeautifulSoup
        import pandas as pd
        
        soup = BeautifulSoup(response.text, 'xml')
        
        # Check if this is an Acknowledgement document (no actual data)
        if soup.find('Acknowledgement_MarketDocument'):
            print("⚠️  Received Acknowledgement document - this means no data available for this period")
            print("Let's try a different date range...")
            
            # Try a few days ago instead
            few_days_ago = today - timedelta(days=3)
            few_days_str = few_days_ago.strftime('%Y%m%d')
            
            retry_params = {
                'documentType': 'A44',
                'in_Domain': '10Y1001A1001A63L',
                'out_Domain': '10Y1001A1001A63L',
                'periodStart': f'{few_days_str}0000',
                'periodEnd': f'{few_days_str}2359',
                'securityToken': ENTSOE_API_TOKEN
            }
            
            print(f"Retrying with {few_days_str}...")
            retry_response = requests.get("https://web-api.tp.entsoe.eu/api", params=retry_params)
            print(f"Retry response: {retry_response.status_code}")
            
            if retry_response.status_code == 200:
                soup = BeautifulSoup(retry_response.text, 'xml')
                if not soup.find('Acknowledgement_MarketDocument'):
                    print("✅ Got actual data on retry!")
                    response = retry_response
                else:
                    print("Still getting Acknowledgement. Let's try a different approach...")
                    raise Exception("No data available")
            else:
                raise Exception("Retry failed")
        
        # Look for actual price data
        time_series = soup.find_all('TimeSeries')
        print(f"Found {len(time_series)} time series in response")
        
        data = []
        for ts in time_series:
            points = ts.find_all('Point')
            print(f"Processing time series with {len(points)} points")
            
            for point in points:
                try:
                    position = int(point.find('position').text)
                    price = float(point.find('price.amount').text)
                    
                    start_time = ts.find('start').text
                    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    actual_dt = start_dt + timedelta(hours=position-1)
                    
                    data.append({
                        'datetime': actual_dt,
                        'price': price
                    })
                except Exception as e:
                    print(f"Error parsing point: {e}")
                    continue
        
        if data:
            real_data = pd.DataFrame(data)
            real_data = real_data.sort_values('datetime').reset_index(drop=True)
            print(f"✅ Parsed {len(real_data)} real price records!")
            print("Sample real data:")
            print(real_data.head())
            print(f"Price range: €{real_data['price'].min():.2f} - €{real_data['price'].max():.2f}/MWh")
            
            # Set this as our data for the rest of the workflow
            data = real_data.copy()
            print("✅ Real data ready for forecasting!")
        else:
            print("❌ No price data found in response")
            print("Full response for debugging:")
            print(response.text[:1000])
            raise Exception("No data found")
            
    except Exception as e:
        print(f"❌ Error parsing real data: {e}")
        print("Falling back to synthetic data...")
        # Don't raise exception, just continue to synthetic data

# If all real data attempts failed, use synthetic data
if response.status_code != 200 or 'data' not in locals():
    print("\n🔧 All real data attempts failed. Using synthetic data for demo...")
    print("This is still valuable for demonstrating the forecasting workflow!")
    
    # Generate synthetic data as fallback
    print("\n📊 Generating synthetic electricity price data...")
    import pandas as pd
    import numpy as np
    
    def generate_synthetic_data(n_samples=8760, start_date='2023-01-01'):
        """Generate synthetic electricity price data for demonstration."""
        dates = pd.date_range(start=start_date, periods=n_samples, freq='H')
        
        # Base price with seasonal patterns
        base_price = 50 + 20 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 365))  # Annual seasonality
        base_price += 10 * np.sin(2 * np.pi * np.arange(n_samples) / 24)  # Daily seasonality
        
        # Add some realistic volatility
        noise = np.random.normal(0, 15, n_samples)
        prices = base_price + noise
        
        # Add some extreme spikes (realistic for electricity markets)
        spike_indices = np.random.choice(n_samples, size=int(0.01 * n_samples), replace=False)
        prices[spike_indices] *= np.random.uniform(2, 5, len(spike_indices))
        
        # Ensure prices are positive
        prices = np.maximum(prices, 5)
        
        data = pd.DataFrame({
            'datetime': dates,
            'price': prices,
            'hour': dates.hour,
            'day_of_week': dates.dayofweek,
            'month': dates.month,
            'year': dates.year
        })
        
        return data
    
    data = generate_synthetic_data()
    print(f"✅ Generated {len(data)} synthetic price records")
    print("Sample data:")
    print(data.head())
    print(f"Price range: €{data['price'].min():.2f} - €{data['price'].max():.2f}/MWh")
```

### 3.2.1 Simple API Test - Let's Debug This Step by Step
```python
# Let's debug the ENTSO-E API step by step
import requests
from datetime import datetime, timedelta

ENTSOE_API_TOKEN = "2c8cd8e0-0a84-4f67-90ba-b79d07ab2667"

print("🔍 Step-by-step API debugging...")

# Step 1: Test with minimal parameters
print("\n1. Testing minimal parameters...")
minimal_params = {
    'documentType': 'A44',
    'securityToken': ENTSOE_API_TOKEN
}

response = requests.get("https://web-api.tp.entsoe.eu/api", params=minimal_params)
print(f"Minimal test: {response.status_code}")
print(f"Response: {response.text[:300]}")

# Step 2: Add domain parameters
print("\n2. Adding domain parameters...")
domain_params = {
    'documentType': 'A44',
    'in_Domain': '10YIT----------',
    'out_Domain': '10YIT----------',
    'securityToken': ENTSOE_API_TOKEN
}

response = requests.get("https://web-api.tp.entsoe.eu/api", params=domain_params)
print(f"Domain test: {response.status_code}")
print(f"Response: {response.text[:300]}")

# Step 3: Add time parameters with today's date
print("\n3. Adding time parameters (today)...")
today = datetime.now()
today_str = today.strftime('%Y%m%d')
today_params = {
    'documentType': 'A44',
    'in_Domain': '10YIT----------',
    'out_Domain': '10YIT----------',
    'periodStart': f'{today_str}0000',
    'periodEnd': f'{today_str}2359',
    'securityToken': ENTSOE_API_TOKEN
}

response = requests.get("https://web-api.tp.entsoe.eu/api", params=today_params)
print(f"Today test: {response.status_code}")
print(f"Response: {response.text[:300]}")

# Step 4: Try yesterday (more likely to have data)
print("\n4. Trying yesterday...")
yesterday = today - timedelta(days=1)
yesterday_str = yesterday.strftime('%Y%m%d')
yesterday_params = {
    'documentType': 'A44',
    'in_Domain': '10YIT----------',
    'out_Domain': '10YIT----------',
    'periodStart': f'{yesterday_str}0000',
    'periodEnd': f'{yesterday_str}2359',
    'securityToken': ENTSOE_API_TOKEN
}

response = requests.get("https://web-api.tp.entsoe.eu/api", params=yesterday_params)
print(f"Yesterday test: {response.status_code}")
if response.status_code == 200:
    print("✅ SUCCESS! API is working!")
    print("Response preview:", response.text[:200])
else:
    print(f"Response: {response.text[:300]}")

# Step 5: If still failing, try Germany
if response.status_code != 200:
    print("\n5. Trying Germany instead of Italy...")
    de_params = {
        'documentType': 'A44',
        'in_Domain': '10YDE----------',
        'out_Domain': '10YDE----------',
        'periodStart': f'{yesterday_str}0000',
        'periodEnd': f'{yesterday_str}2359',
        'securityToken': ENTSOE_API_TOKEN
    }
    
    response = requests.get("https://web-api.tp.entsoe.eu/api", params=de_params)
    print(f"Germany test: {response.status_code}")
    if response.status_code == 200:
        print("✅ SUCCESS with Germany!")
        print("Response preview:", response.text[:200])
    else:
        print(f"Germany response: {response.text[:300]}")
        print("\n🔧 API seems to have issues. Let's proceed with synthetic data for the demo.")
        
        # Generate synthetic data as fallback
        print("\n📊 Generating synthetic electricity price data...")
        import pandas as pd
        import numpy as np
        
        def generate_synthetic_data(n_samples=8760, start_date='2023-01-01'):
            """Generate synthetic electricity price data for demonstration."""
            dates = pd.date_range(start=start_date, periods=n_samples, freq='H')
            
            # Base price with seasonal patterns
            base_price = 50 + 20 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 365))  # Annual seasonality
            base_price += 10 * np.sin(2 * np.pi * np.arange(n_samples) / 24)  # Daily seasonality
            
            # Add some realistic volatility
            noise = np.random.normal(0, 15, n_samples)
            prices = base_price + noise
            
            # Add some extreme spikes (realistic for electricity markets)
            spike_indices = np.random.choice(n_samples, size=int(0.01 * n_samples), replace=False)
            prices[spike_indices] *= np.random.uniform(2, 5, len(spike_indices))
            
            # Ensure prices are positive
            prices = np.maximum(prices, 5)
            
            data = pd.DataFrame({
                'datetime': dates,
                'price': prices,
                'hour': dates.hour,
                'day_of_week': dates.dayofweek,
                'month': dates.month,
                'year': dates.year
            })
            
            return data
        
        synthetic_data = generate_synthetic_data()
        print(f"✅ Generated {len(synthetic_data)} synthetic price records")
        print("Sample data:")
        print(synthetic_data.head())
        print(f"Price range: €{synthetic_data['price'].min():.2f} - €{synthetic_data['price'].max():.2f}/MWh")
```

### 3.2.1 Quick Fix for Date Format Issue
```python
# Quick fix for the 400 error - ENTSO-E API expects specific date format
print("Testing with correct date format...")

ENTSOE_API_TOKEN = "2c8cd8e0-0a84-4f67-90ba-b79d07ab2667"

# Test with correct date format (YYYYMMDDHHMM)
test_params = {
    'documentType': 'A44',
    'in_Domain': '10YIT----------',
    'out_Domain': '10YIT----------',
    'periodStart': '202401010000',  # 2024-01-01 00:00
    'periodEnd': '202401012359',    # 2024-01-01 23:59
    'securityToken': ENTSOE_API_TOKEN
}

response = requests.get("https://web-api.tp.entsoe.eu/api", params=test_params)
print(f"API Response Status: {response.status_code}")

if response.status_code == 200:
    print("✓ API is working with correct date format!")
    print("Response preview:", response.text[:200])
elif response.status_code == 400:
    print("Still getting 400 error. Let's try a different approach...")
    print("Full Response:", response.text)
    
    # Try with a more recent date range (maybe 2024-01-01 is too old)
    print("\nTrying with more recent date...")
    recent_params = {
        'documentType': 'A44',
        'in_Domain': '10YIT----------',
        'out_Domain': '10YIT----------',
        'periodStart': '202412010000',  # 2024-12-01 00:00
        'periodEnd': '202412012359',    # 2024-12-01 23:59
        'securityToken': ENTSOE_API_TOKEN
    }
    
    response2 = requests.get("https://web-api.tp.entsoe.eu/api", params=recent_params)
    print(f"Recent date response: {response2.status_code}")
    if response2.status_code == 200:
        print("✓ API works with recent date!")
        print("Response preview:", response2.text[:200])
    else:
        print("Still 400. Let's try Germany instead...")
        
        # Try Germany (DE) instead of Italy (IT)
        de_params = {
            'documentType': 'A44',
            'in_Domain': '10YDE----------',
            'out_Domain': '10YDE----------',
            'periodStart': '202412010000',
            'periodEnd': '202412012359',
            'securityToken': ENTSOE_API_TOKEN
        }
        
        response3 = requests.get("https://web-api.tp.entsoe.eu/api", params=de_params)
        print(f"Germany response: {response3.status_code}")
        if response3.status_code == 200:
            print("✓ API works with Germany!")
            print("Response preview:", response3.text[:200])
        else:
            print("Still having issues. Full response:", response3.text[:500])
            print("\n🔧 Let's use synthetic data for now and continue with the demo...")
            print("The API token is working, but there might be data availability issues.")
else:
    print(f"Response: {response.status_code}")
    print("Response text:", response.text[:200])
```

### 3.3 Test Your API Token
```python
# Test your ENTSO-E API token
ENTSOE_API_TOKEN = "2c8cd8e0-0a84-4f67-90ba-b79d07ab2667"

print("Testing ENTSO-E API token...")
downloader = ENTSOEDownloader(api_token=ENTSOE_API_TOKEN)

# Test API connection manually (since test_api_connection might not be available yet)
print("Testing API connection...")
import requests

try:
    # Test the API with a simple request
    test_params = {
        'documentType': 'A44',
        'in_Domain': '10YIT----------',
        'out_Domain': '10YIT----------',
        'periodStart': '202401010000',
        'periodEnd': '202401012359',
        'securityToken': ENTSOE_API_TOKEN
    }
    
    response = requests.get("https://web-api.tp.entsoe.eu/api", params=test_params)
    print(f"API Response Status: {response.status_code}")
    
    if response.status_code == 200:
        print("✓ API token is valid!")
        
        # Now try to download a small sample of data
        print("Downloading test data...")
        try:
            test_data = downloader.download_price_data(
                country='IT',
                start_date='2024-01-01',
                end_date='2024-01-02',  # Just one day for testing
                data_type='day_ahead'
            )
            
            if not test_data.empty:
                print("✓ Data download successful!")
                print(f"Downloaded {len(test_data)} test records")
                print("Sample data:")
                print(test_data.head())
            else:
                print("✗ No data returned - API may be working but no data available for this period")
                
        except Exception as e:
            print(f"✗ Data download failed: {e}")
            
    elif response.status_code == 401:
        print("✗ API token is invalid or not activated")
        print("Response:", response.text[:200])
        print("Possible reasons:")
        print("1. Token not activated yet (wait 3 business days)")
        print("2. Token is invalid")
        print("3. Token format is incorrect")
    else:
        print(f"✗ Unexpected response: {response.status_code}")
        print("Response:", response.text[:200])
        
except Exception as e:
    print(f"✗ API test failed: {e}")
    print("Possible reasons:")
    print("1. Network connectivity issues")
    print("2. ENTSO-E API is temporarily unavailable")
    print("3. Token not activated yet")
```

### 4. Data Collection (Choose One)

#### Option A: Real Data (Requires API Token)
```python
# Set your ENTSO-E API token
ENTSOE_API_TOKEN = "2c8cd8e0-0a84-4f67-90ba-b79d07ab2667"  # Your actual token

print("Downloading real electricity price data from ENTSO-E...")
downloader = ENTSOEDownloader(api_token=ENTSOE_API_TOKEN)

try:
    data = downloader.download_price_data(
        country='IT',  # Italy - you can change to other countries
        start_date='2023-01-01',
        end_date='2024-01-01',
        data_type='day_ahead'
    )
    
    if not data.empty:
        print(f"✓ Downloaded {len(data)} real price records")
        print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")
        print(f"Price range: €{data['price'].min():.2f} - €{data['price'].max():.2f}/MWh")
        print("First few rows:")
        print(data.head())
    else:
        print("✗ No data downloaded - trying synthetic data instead")
        # Fall back to synthetic data
        data = generate_synthetic_data()
        print(f"Generated {len(data)} synthetic price records")
        
except Exception as e:
    print(f"✗ Error downloading real data: {e}")
    print("Falling back to synthetic data...")
    # Fall back to synthetic data
    data = generate_synthetic_data()
    print(f"Generated {len(data)} synthetic price records")
```

#### Option B: Synthetic Data (No API Required)
```python
# Generate synthetic data
def generate_synthetic_data(n_samples=8760, start_date='2023-01-01'):
    np.random.seed(42)
    dates = pd.date_range(start_date, periods=n_samples, freq='H')
    
    base_price = 50
    daily_pattern = 15 * np.sin(2 * np.pi * np.arange(n_samples) / 24 - np.pi/2)
    weekly_pattern = 5 * (dates.dayofweek < 5).astype(int)
    monthly_pattern = 10 * np.sin(2 * np.pi * dates.month / 12)
    trend = 0.001 * np.arange(n_samples)
    noise = np.random.normal(0, 3, n_samples)
    spikes = np.random.binomial(1, 0.02, n_samples) * np.random.exponential(20, n_samples)
    
    price = base_price + daily_pattern + weekly_pattern + monthly_pattern + trend + noise + spikes
    price = np.maximum(price, 0)
    
    return pd.DataFrame({
        'datetime': dates,
        'price': price,
        'hour': dates.hour,
        'day_of_week': dates.dayofweek,
        'month': dates.month,
        'is_weekend': (dates.dayofweek >= 5).astype(int)
    })

data = generate_synthetic_data()
print(f"Generated {len(data)} synthetic price records")
```

### 5. Data Preprocessing
```python
# Preprocess data
preprocessor = DataPreprocessor()

# Check data structure first
print("Data columns:", data.columns.tolist())
print("Data shape:", data.shape)
print("First few rows:")
print(data.head())

# Set datetime as index (check if column exists)
if 'datetime' in data.columns:
    data = data.set_index('datetime')
    print("Set 'datetime' as index")
elif 'date' in data.columns:
    data = data.set_index('date')
    print("Set 'date' as index")
else:
    print("Available columns:", data.columns.tolist())
    print("Please check your data structure")

clean_data = preprocessor.clean_price_data(data)
features_df = preprocessor.engineer_features(clean_data)

# Handle missing values (important for ML models)
print("Handling missing values...")
print(f"Missing values before: {features_df.isnull().sum().sum()}")

# Fill missing values with forward fill, then backward fill
features_df = features_df.fillna(method='ffill').fillna(method='bfill')

# If still missing values, fill with mean
features_df = features_df.fillna(features_df.mean())

print(f"Missing values after: {features_df.isnull().sum().sum()}")

# Prepare training data
X_train, X_test, y_train, y_test = preprocessor.prepare_training_data(
    target_column='price',
    test_size=0.2
)

print(f"Training data: {X_train.shape}")
print(f"Test data: {X_test.shape}")
print(f"Training missing values: {X_train.isnull().sum().sum()}")
print(f"Test missing values: {X_test.isnull().sum().sum()}")
```

### 6. Train Models
```python
# Train baseline models
baseline_models = BaselineModels()
baseline_models.train_all(X_train, y_train)
baseline_predictions = baseline_models.predict_all(X_test)
baseline_results = baseline_models.evaluate_all(y_test, baseline_predictions)

# Train ML models
ml_models = MLModels()
ml_models.train_all(X_train, y_train, tune_hyperparameters=False)
ml_predictions = ml_models.predict_all(X_test)
ml_results = ml_models.evaluate_all(y_test, ml_predictions)

print("Models trained successfully!")
```

### 6.1 Quick Fix for Data Issues (if you get errors)

#### Fix 1: Column Name Issues
```python
# If you get KeyError about 'datetime' column, run this first
print("Checking data structure...")
print("Data columns:", data.columns.tolist())
print("Data shape:", data.shape)

# Fix column name if needed
if 'datetime' not in data.columns and 'date' in data.columns:
    data = data.rename(columns={'date': 'datetime'})
    print("Renamed 'date' to 'datetime'")

# Set index
data = data.set_index('datetime')
print("Data index set successfully")
```

#### Fix 2: NaN Values
```python
# If you get NaN errors, run this cell to fix the data
print("Fixing NaN values in training data...")
print(f"X_train NaN count: {X_train.isnull().sum().sum()}")
print(f"X_test NaN count: {X_test.isnull().sum().sum()}")

# Fill missing values
X_train = X_train.fillna(method='ffill').fillna(method='bfill').fillna(X_train.mean())
X_test = X_test.fillna(method='ffill').fillna(method='bfill').fillna(X_train.mean())

print(f"After fixing - X_train NaN count: {X_train.isnull().sum().sum()}")
print(f"After fixing - X_test NaN count: {X_test.isnull().sum().sum()}")

# Now retrain the models
print("Retraining models with clean data...")
ml_models = MLModels()
ml_models.train_all(X_train, y_train, tune_hyperparameters=False)
ml_predictions = ml_models.predict_all(X_test)
ml_results = ml_models.evaluate_all(y_test, ml_predictions)

print("Models retrained successfully!")
```

### 7. Evaluate and Visualize
```python
# Comprehensive evaluation
evaluator = EvaluationMetrics()
all_predictions = {**baseline_predictions, **ml_predictions}
all_results = {}

for model_name, pred in all_predictions.items():
    metrics = evaluator.calculate_all_metrics(y_test, pred, model_name)
    all_results[model_name] = metrics

comparison_df = evaluator.compare_models(all_results)
print("Model Performance Comparison:")
print(comparison_df.round(4))

# Create interactive plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test.values, 
                mode='lines', name='Actual', line=dict(width=2, color='black')))

for model_name, pred in all_predictions.items():
    fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=pred, 
                            mode='lines', name=model_name))

fig.update_layout(title='Electricity Price Predictions vs Actual',
                  xaxis_title='Time Index', yaxis_title='Price (€/MWh)')
fig.show()
```

### 8. Business Analysis
```python
# Business impact analysis
best_model_name = comparison_df.index[0]
best_pred = all_predictions[best_model_name]

total_cost_error = np.sum(np.abs(y_test - best_pred))
avg_cost_error = total_cost_error / len(y_test)

print(f"Best Model: {best_model_name}")
print(f"RMSE: {comparison_df.loc[best_model_name, 'rmse']:.2f} €/MWh")
print(f"MAE: {comparison_df.loc[best_model_name, 'mae']:.2f} €/MWh")
print(f"Total cost error: €{total_cost_error:.2f}")
print(f"Average cost error per hour: €{avg_cost_error:.2f}")
```

## Colab Advantages

1. **Free GPU Access**: Enable GPU in Runtime > Change runtime type
2. **Pre-installed Libraries**: Most ML libraries already available
3. **Interactive Visualizations**: Plotly works great in Colab
4. **Easy Sharing**: Share notebooks with others
5. **Cloud Storage**: Save results to Google Drive

## Tips for Colab

1. **Enable GPU**: Runtime > Change runtime type > GPU
2. **Save Progress**: Mount Google Drive to save results
3. **Session Timeout**: Colab sessions timeout after inactivity
4. **Memory Limits**: Large datasets may hit memory limits
5. **Install Once**: Run setup cell only once per session

## Next Steps

1. **Get API Token**: Register at https://transparency.entsoe.eu/
2. **Experiment**: Try different models and parameters
3. **Add Features**: Include weather data, demand forecasts
4. **Deploy**: Export models for production use
5. **Share**: Share your Colab notebook with others

## Troubleshooting

- **Import Errors**: Make sure you're in the correct directory
- **Memory Issues**: Reduce dataset size or use smaller models
- **Timeout**: Save work frequently and restart if needed
- **API Errors**: Check your ENTSO-E token is valid
