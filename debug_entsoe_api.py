#!/usr/bin/env python3
"""
Debug script for ENTSO-E API testing.
Run this locally to debug API issues with better error handling and logging.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import json
import time

# Your API token
ENTSOE_API_TOKEN = "2c8cd8e0-0a84-4f67-90ba-b79d07ab2667"

def test_entsoe_api():
    """Test ENTSO-E API with comprehensive debugging."""
    
    print("🔍 ENTSO-E API Debugging Tool")
    print("=" * 50)
    
    # Test different countries and date ranges
    countries = {
        'Germany': '10Y1001A1001A63L',
        'France': '10YFR-RTE------C', 
        'Netherlands': '10YNL----------L',
        'Italy': '10YIT----------',
        'Spain': '10YES-REE------0'
    }
    
    # Test different date ranges
    today = datetime.now()
    date_ranges = {
        'Yesterday': today - timedelta(days=1),
        '1 week ago': today - timedelta(days=7),
        '2 weeks ago': today - timedelta(days=14),
        '1 month ago': today - timedelta(days=30),
        '2 months ago': today - timedelta(days=60),
        '3 months ago': today - timedelta(days=90)
    }
    
    successful_data = None
    
    for country_name, domain_code in countries.items():
        print(f"\n🌍 Testing {country_name} (Domain: {domain_code})")
        print("-" * 40)
        
        for period_name, test_date in date_ranges.items():
            date_str = test_date.strftime('%Y%m%d')
            
            print(f"  📅 {period_name} ({date_str})...")
            
            # Test parameters
            params = {
                'documentType': 'A44',
                'in_Domain': domain_code,
                'out_Domain': domain_code,
                'periodStart': f'{date_str}0000',
                'periodEnd': f'{date_str}2359',
                'securityToken': ENTSOE_API_TOKEN
            }
            
            try:
                # Make API request
                response = requests.get("https://web-api.tp.entsoe.eu/api", params=params, timeout=30)
                
                print(f"    Status: {response.status_code}")
                
                if response.status_code == 200:
                    # Parse XML response
                    soup = BeautifulSoup(response.text, 'xml')
                    
                    # Check if it's an Acknowledgement document
                    if soup.find('Acknowledgement_MarketDocument'):
                        print(f"    ❌ Acknowledgement document (no data)")
                        continue
                    
                    # Look for actual price data
                    time_series = soup.find_all('TimeSeries')
                    print(f"    📊 Found {len(time_series)} time series")
                    
                    if time_series:
                        # Try to parse the data
                        data = parse_price_data(soup)
                        
                        if data and len(data) > 0:
                            print(f"    ✅ SUCCESS! Parsed {len(data)} price records")
                            print(f"    💰 Price range: €{data['price'].min():.2f} - €{data['price'].max():.2f}/MWh")
                            print(f"    📅 Date range: {data['datetime'].min()} to {data['datetime'].max()}")
                            
                            # Save successful data
                            successful_data = data
                            
                            # Show sample data
                            print("    📋 Sample data:")
                            print(data.head().to_string(index=False))
                            
                            return data, country_name, period_name
                        else:
                            print(f"    ❌ No price data found in time series")
                    else:
                        print(f"    ❌ No time series found")
                        
                elif response.status_code == 400:
                    print(f"    ❌ Bad Request - checking response...")
                    soup = BeautifulSoup(response.text, 'xml')
                    if soup.find('Acknowledgement_MarketDocument'):
                        print(f"    📝 Acknowledgement document (no data for this period)")
                    else:
                        print(f"    📝 Response: {response.text[:200]}...")
                        
                elif response.status_code == 401:
                    print(f"    ❌ Unauthorized - API token issue")
                    return None, None, None
                    
                else:
                    print(f"    ❌ Error {response.status_code}: {response.text[:200]}...")
                
                # Rate limiting
                time.sleep(1)
                
            except requests.exceptions.Timeout:
                print(f"    ⏰ Timeout - request took too long")
            except requests.exceptions.RequestException as e:
                print(f"    ❌ Request error: {e}")
            except Exception as e:
                print(f"    ❌ Parsing error: {e}")
    
    print(f"\n🔧 No real data found for any country/date combination")
    print("Generating synthetic data for demonstration...")
    
    return generate_synthetic_data(), "Synthetic", "Generated"

def parse_price_data(soup):
    """Parse price data from XML response."""
    try:
        time_series = soup.find_all('TimeSeries')
        data = []
        
        for ts in time_series:
            points = ts.find_all('Point')
            
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
                    print(f"      Error parsing point: {e}")
                    continue
        
        if data:
            df = pd.DataFrame(data)
            df = df.sort_values('datetime').reset_index(drop=True)
            return df
        else:
            return None
            
    except Exception as e:
        print(f"Error parsing price data: {e}")
        return None

def generate_synthetic_data(n_samples=8760, start_date='2023-01-01'):
    """Generate synthetic electricity price data."""
    print("📊 Generating synthetic electricity price data...")
    
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
    
    print(f"✅ Generated {len(data)} synthetic price records")
    print(f"💰 Price range: €{data['price'].min():.2f} - €{data['price'].max():.2f}/MWh")
    
    return data

def main():
    """Main debugging function."""
    print("Starting ENTSO-E API debugging...")
    
    # Test the API
    data, country, period = test_entsoe_api()
    
    if data is not None:
        print(f"\n🎉 SUCCESS!")
        print(f"Country: {country}")
        print(f"Period: {period}")
        print(f"Records: {len(data)}")
        print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")
        print(f"Price range: €{data['price'].min():.2f} - €{data['price'].max():.2f}/MWh")
        
        # Save the data
        filename = f"electricity_prices_{country.lower().replace(' ', '_')}_{period.replace(' ', '_')}.csv"
        data.to_csv(filename, index=False)
        print(f"💾 Data saved to: {filename}")
        
        return data
    else:
        print("\n❌ No data obtained")
        return None

if __name__ == "__main__":
    data = main()
