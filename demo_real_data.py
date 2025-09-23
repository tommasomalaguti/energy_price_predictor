#!/usr/bin/env python3
"""
Demo script using real electricity price data from France.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append('src')

def load_real_data():
    """Load the real electricity price data we obtained."""
    try:
        data = pd.read_csv('electricity_prices_france_Yesterday.csv')
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.set_index('datetime')
        print(f"✅ Loaded real data: {len(data)} records")
        print(f"📅 Date range: {data.index.min()} to {data.index.max()}")
        print(f"💰 Price range: €{data['price'].min():.2f} - €{data['price'].max():.2f}/MWh")
        return data
    except FileNotFoundError:
        print("❌ Real data file not found. Run debug_entsoe_api.py first.")
        return None

def analyze_real_data(data):
    """Analyze the real electricity price data."""
    print("\n📊 Real Data Analysis")
    print("=" * 40)
    
    # Basic statistics
    print(f"Records: {len(data)}")
    print(f"Mean price: €{data['price'].mean():.2f}/MWh")
    print(f"Median price: €{data['price'].median():.2f}/MWh")
    print(f"Std deviation: €{data['price'].std():.2f}/MWh")
    print(f"Min price: €{data['price'].min():.2f}/MWh")
    print(f"Max price: €{data['price'].max():.2f}/MWh")
    
    # Price patterns
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    
    print(f"\n🕐 Hourly patterns:")
    hourly_stats = data.groupby('hour')['price'].agg(['mean', 'std']).round(2)
    print(hourly_stats.head(10))
    
    # Check for negative prices (common in electricity markets)
    negative_prices = data[data['price'] < 0]
    if len(negative_prices) > 0:
        print(f"\n⚡ Negative prices: {len(negative_prices)} records")
        print("This is normal in electricity markets (excess renewable energy)")
    
    return data

def create_visualizations(data):
    """Create visualizations of the real data."""
    print("\n📈 Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Real French Electricity Price Analysis', fontsize=16, fontweight='bold')
    
    # Time series plot
    axes[0, 0].plot(data.index, data['price'], linewidth=2, color='blue', alpha=0.7)
    axes[0, 0].set_title('Electricity Prices Over Time')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Price (€/MWh)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Price distribution
    axes[0, 1].hist(data['price'], bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_title('Price Distribution')
    axes[0, 1].set_xlabel('Price (€/MWh)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Hourly patterns
    hourly_avg = data.groupby('hour')['price'].mean()
    axes[1, 0].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, color='red')
    axes[1, 0].set_title('Average Price by Hour of Day')
    axes[1, 0].set_xlabel('Hour of Day')
    axes[1, 0].set_ylabel('Average Price (€/MWh)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Box plot by hour
    data['hour'] = data.index.hour
    hourly_data = [data[data['hour'] == h]['price'].values for h in range(24)]
    axes[1, 1].boxplot(hourly_data, positions=range(24))
    axes[1, 1].set_title('Price Distribution by Hour')
    axes[1, 1].set_xlabel('Hour of Day')
    axes[1, 1].set_ylabel('Price (€/MWh)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('real_data_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Visualizations saved to 'real_data_analysis.png'")
    
    return fig

def demonstrate_forecasting(data):
    """Demonstrate simple forecasting with real data."""
    print("\n🔮 Simple Forecasting Demo")
    print("=" * 40)
    
    # Simple moving average forecast
    window = 6  # 6-hour moving average
    data['ma_forecast'] = data['price'].rolling(window=window).mean().shift(1)
    
    # Calculate forecast accuracy
    valid_forecasts = data.dropna()
    if len(valid_forecasts) > 0:
        mae = np.mean(np.abs(valid_forecasts['price'] - valid_forecasts['ma_forecast']))
        mape = np.mean(np.abs((valid_forecasts['price'] - valid_forecasts['ma_forecast']) / valid_forecasts['price'])) * 100
        
        print(f"📊 Moving Average Forecast (6-hour window):")
        print(f"   MAE: €{mae:.2f}/MWh")
        print(f"   MAPE: {mape:.1f}%")
        
        # Show some forecast examples
        print(f"\n📋 Sample Forecasts:")
        sample = valid_forecasts[['price', 'ma_forecast']].head(10)
        sample['error'] = sample['price'] - sample['ma_forecast']
        print(sample.round(2))
    
    return data

def main():
    """Main demo function."""
    print("🇫🇷 French Electricity Price Forecasting Demo")
    print("=" * 50)
    
    # Load real data
    data = load_real_data()
    if data is None:
        return
    
    # Analyze the data
    data = analyze_real_data(data)
    
    # Create visualizations
    fig = create_visualizations(data)
    
    # Demonstrate forecasting
    data = demonstrate_forecasting(data)
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"📁 Files created:")
    print(f"   - electricity_prices_france_Yesterday.csv (raw data)")
    print(f"   - real_data_analysis.png (visualizations)")
    
    print(f"\n💡 Key Insights:")
    print(f"   - Real French electricity prices range from €{data['price'].min():.2f} to €{data['price'].max():.2f}/MWh")
    print(f"   - Average price: €{data['price'].mean():.2f}/MWh")
    print(f"   - Data shows typical electricity market patterns")
    print(f"   - Ready for advanced forecasting models!")

if __name__ == "__main__":
    main()
