# Electricity Price Forecasting

A comprehensive Python project for forecasting day-ahead electricity market prices in Europe using public ENTSO-E data and machine learning.

## Project Overview

This project demonstrates solid data science workflow for electricity price forecasting, comparing different forecasting models (statistical, ML, and deep learning) and analyzing their usefulness for industrial buyers.

## Features

- **Data Collection**: Automated download and preprocessing of ENTSO-E hourly market price data
- **Feature Engineering**: Time-based features, lag features, weather integration, and demand patterns
- **Multiple Models**: Baselines, classical ML, and time series forecasting models
- **Comprehensive Evaluation**: RMSE, MAE, MAPE metrics with visual analysis
- **Industrial Focus**: Practical impact analysis for energy buyers

## Project Structure

```
energy_price_predictor/
├── data/                    # Raw and processed data
│   ├── raw/                # Raw ENTSO-E data
│   ├── processed/          # Cleaned and engineered data
│   └── external/           # Weather and other external data
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Source code modules
│   ├── data/              # Data collection and preprocessing
│   ├── features/          # Feature engineering
│   ├── models/            # Forecasting models
│   ├── evaluation/        # Model evaluation and metrics
│   └── utils/             # Utility functions
├── config/                 # Configuration files
├── results/               # Model results and visualizations
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd energy_price_predictor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

1. **Download Data**: Run the data collection script for your target country
```python
from src.data.entsoe_downloader import ENTSOEDownloader
downloader = ENTSOEDownloader()
downloader.download_price_data(country='IT', start_date='2022-01-01')
```

2. **Run Analysis**: Use the provided Jupyter notebooks for exploratory analysis and modeling

3. **Generate Forecasts**: Use the modeling pipeline to train and evaluate different models

## Data Sources

- **ENTSO-E**: European electricity market price data
- **Weather APIs**: Temperature, solar, and wind data
- **Holiday Data**: National holidays and special events

## Models Implemented

### Baselines
- Naive (last observation)
- Historical mean
- Seasonal naive

### Classical ML
- Linear Regression
- Ridge/Lasso Regression
- Random Forest
- XGBoost

### Time Series
- ARIMA/SARIMAX
- Prophet
- LSTM/GRU (optional)

## Evaluation Metrics

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **Directional Accuracy**: Percentage of correct price direction predictions

## Results and Analysis

The project provides comprehensive analysis of model performance with focus on practical implications for industrial energy buyers, including cost estimation accuracy and risk assessment.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ENTSO-E for providing open electricity market data
- European weather services for meteorological data
- The open-source Python data science community
