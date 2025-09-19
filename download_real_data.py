#!/usr/bin/env python3
"""
Simple script to download real electricity price data from ENTSO-E.

This script shows how to download actual electricity market data
without running the full forecasting pipeline.
"""

import sys
import os
sys.path.append('src')

from data.entsoe_downloader import ENTSOEDownloader
import pandas as pd


def main():
    """Download real electricity price data."""
    print("=== DOWNLOADING REAL ELECTRICITY PRICE DATA ===\n")
    
    # Check if API token is available
    api_token = os.getenv('ENTSOE_API_TOKEN')
    if not api_token:
        print("ERROR: No ENTSO-E API token found!")
        print("\nTo get a token:")
        print("1. Go to https://transparency.entsoe.eu/")
        print("2. Register for a free account")
        print("3. Get your API token")
        print("4. Set it as an environment variable:")
        print("   export ENTSOE_API_TOKEN='your_token_here'")
        return
    
    print(f"Using ENTSO-E API token: {api_token[:10]}...")
    
    # Initialize downloader
    downloader = ENTSOEDownloader(api_token=api_token)
    
    # Show available countries
    print("\nAvailable countries:")
    countries = downloader.get_available_countries()
    for code, name in list(countries.items())[:10]:  # Show first 10
        print(f"  {code}: {name}")
    print("  ... and more")
    
    # Download data for different countries
    countries_to_download = ['IT', 'DE', 'FR', 'ES']  # Italy, Germany, France, Spain
    
    for country in countries_to_download:
        print(f"\n--- Downloading data for {country} ({countries[country]}) ---")
        
        try:
            # Download last 6 months of data
            data = downloader.download_price_data(
                country=country,
                start_date='2024-01-01',
                end_date='2024-07-01',
                data_type='day_ahead',
                save_path=f'data/raw/{country.lower()}_prices_2024.csv'
            )
            
            if not data.empty:
                print(f"   ✓ Downloaded {len(data)} records")
                print(f"   Date range: {data['datetime'].min()} to {data['datetime'].max()}")
                print(f"   Price range: €{data['price'].min():.2f} - €{data['price'].max():.2f}/MWh")
                print(f"   Average price: €{data['price'].mean():.2f}/MWh")
                print(f"   Saved to: data/raw/{country.lower()}_prices_2024.csv")
            else:
                print(f"   ✗ No data downloaded for {country}")
                
        except Exception as e:
            print(f"   ✗ Error downloading {country}: {e}")
    
    print("\n=== DOWNLOAD COMPLETE ===")
    print("You can now use this real data with the forecasting models!")
    print("\nTo use the real data:")
    print("1. Run: python real_data_example.py")
    print("2. Or use the data in your own analysis")


if __name__ == "__main__":
    main()
