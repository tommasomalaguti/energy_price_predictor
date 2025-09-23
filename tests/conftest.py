"""
Pytest configuration and shared fixtures for electricity price forecasting tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture
def sample_price_data():
    """Generate sample electricity price data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create time index
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Generate realistic price data with patterns
    base_price = 50
    daily_pattern = 15 * np.sin(2 * np.pi * np.arange(n_samples) / 24)
    weekly_pattern = 5 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 7))
    noise = np.random.normal(0, 5, n_samples)
    
    prices = base_price + daily_pattern + weekly_pattern + noise
    
    data = pd.DataFrame({
        'datetime': dates,
        'price': prices
    })
    data.set_index('datetime', inplace=True)
    
    return data

@pytest.fixture
def sample_features():
    """Generate sample feature data for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Generate random features
    features = np.random.randn(n_samples, n_features)
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    data = pd.DataFrame(features, columns=feature_names)
    
    return data

@pytest.fixture
def sample_train_test_data(sample_price_data, sample_features):
    """Generate train/test split for testing."""
    from sklearn.model_selection import train_test_split
    
    # Use last 20% for testing
    split_idx = int(len(sample_price_data) * 0.8)
    
    X_train = sample_features[:split_idx]
    X_test = sample_features[split_idx:]
    y_train = sample_price_data['price'][:split_idx]
    y_test = sample_price_data['price'][split_idx:]
    
    return X_train, X_test, y_train, y_test

@pytest.fixture
def mock_entsoe_response():
    """Mock ENTSO-E API response for testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
    <Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0">
        <mRID>test123</mRID>
        <revisionNumber>1</revisionNumber>
        <type>A44</type>
        <sender_MarketParticipant.mRID codingScheme="A01">10X1001A1001A450</sender_MarketParticipant.mRID>
        <sender_MarketParticipant.marketRole.mRID codingScheme="A01">A01</sender_MarketParticipant.marketRole.mRID>
        <receiver_MarketParticipant.mRID codingScheme="A01">10X1001A1001A450</receiver_MarketParticipant.mRID>
        <receiver_MarketParticipant.marketRole.mRID codingScheme="A01">A01</receiver_MarketParticipant.marketRole.mRID>
        <createdDateTime>2023-01-01T00:00:00Z</createdDateTime>
        <period.timeInterval>
            <start>2023-01-01T00:00Z</start>
            <end>2023-01-01T23:59Z</end>
        </period.timeInterval>
        <TimeSeries>
            <mRID>1</mRID>
            <businessType>B01</businessType>
            <in_Domain.mRID codingScheme="A01">10Y1001A1001A63L</in_Domain.mRID>
            <out_Domain.mRID codingScheme="A01">10Y1001A1001A63L</out_Domain.mRID>
            <quantity_Measure_Unit.name>MWH</quantity_Measure_Unit.name>
            <curveType>A01</curveType>
            <Period>
                <timeInterval>
                    <start>2023-01-01T00:00Z</start>
                    <end>2023-01-01T23:59Z</end>
                </timeInterval>
                <resolution>PT1H</resolution>
                <Point>
                    <position>1</position>
                    <price.amount>45.50</price.amount>
                </Point>
                <Point>
                    <position>2</position>
                    <price.amount>42.30</price.amount>
                </Point>
            </Period>
        </TimeSeries>
    </Publication_MarketDocument>"""
