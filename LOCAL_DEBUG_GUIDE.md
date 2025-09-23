# Local Debugging Guide for ENTSO-E API

This guide helps you debug the ENTSO-E API issues locally with better error handling and logging.

## Quick Start

### 1. Set up the environment
```bash
# Install Python dependencies
python setup_local.py

# Or manually install requirements
pip install -r requirements_local.txt
```

### 2. Run the debug script
```bash
python debug_entsoe_api.py
```

## What the Debug Script Does

The `debug_entsoe_api.py` script will:

1. **Test multiple countries**: Germany, France, Netherlands, Italy, Spain
2. **Test multiple date ranges**: Yesterday, 1 week ago, 2 weeks ago, 1 month ago, 2 months ago, 3 months ago
3. **Provide detailed logging**: Shows exactly what's happening with each request
4. **Parse XML responses**: Attempts to extract actual price data
5. **Save successful data**: Exports any real data found to CSV
6. **Fallback to synthetic**: Generates realistic synthetic data if no real data is found

## Expected Output

```
🔍 ENTSO-E API Debugging Tool
==================================================

🌍 Testing Germany (Domain: 10Y1001A1001A63L)
----------------------------------------
  📅 Yesterday (20250922)...
    Status: 200
    ❌ Acknowledgement document (no data)
  📅 1 week ago (20250916)...
    Status: 200
    ❌ Acknowledgement document (no data)
  📅 2 weeks ago (20250909)...
    Status: 200
    📊 Found 1 time series
    ✅ SUCCESS! Parsed 24 price records
    💰 Price range: €45.23 - €78.91/MWh
    📅 Date range: 2025-09-09 00:00:00 to 2025-09-09 23:00:00
    📋 Sample data:
    datetime  price
    2025-09-09 00:00:00  52.34
    2025-09-09 01:00:00  48.76
    ...

🎉 SUCCESS!
Country: Germany
Period: 2 weeks ago
Records: 24
Date range: 2025-09-09 00:00:00 to 2025-09-09 23:00:00
Price range: €45.23 - €78.91/MWh
💾 Data saved to: electricity_prices_germany_2_weeks_ago.csv
```

## Troubleshooting

### If you get "No data found":
- The API is working (200 responses) but no data is available for the tested periods
- This is common with ENTSO-E - data might not be available for recent periods
- The script will automatically generate synthetic data as a fallback

### If you get import errors:
- Run `python setup_local.py` to install all dependencies
- Or manually install: `pip install -r requirements_local.txt`

### If you get API errors:
- Check your API token is correct
- Verify your internet connection
- The script includes rate limiting to avoid overwhelming the API

## Next Steps

Once you have data (real or synthetic):

1. **Run the full forecasting workflow**:
   ```bash
   python real_data_example.py
   ```

2. **Or use the Jupyter notebook**:
   ```bash
   jupyter notebook notebooks/01_electricity_price_forecasting_demo.ipynb
   ```

3. **Debug specific issues**:
   - Modify `debug_entsoe_api.py` to test different parameters
   - Add more countries or date ranges
   - Check the XML responses manually

## File Structure

```
energy_price_predictor/
├── debug_entsoe_api.py          # Main debugging script
├── setup_local.py               # Local setup script
├── requirements_local.txt       # Local dependencies
├── LOCAL_DEBUG_GUIDE.md         # This guide
├── src/                         # Source code
├── data/                        # Data directories
└── notebooks/                   # Jupyter notebooks
```

## API Token

Make sure your ENTSO-E API token is set in `debug_entsoe_api.py`:
```python
ENTSOE_API_TOKEN = "your_token_here"
```

The token should be the one you received from ENTSO-E: `2c8cd8e0-0a84-4f67-90ba-b79d07ab2667`
