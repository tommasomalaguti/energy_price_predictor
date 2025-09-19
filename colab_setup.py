#!/usr/bin/env python3
"""
Google Colab setup script for electricity price forecasting.

Run this in a Colab cell to set up the environment and install dependencies.
"""

# Install required packages
import subprocess
import sys

def install_packages():
    """Install required packages for Colab."""
    packages = [
        'xgboost',
        'lightgbm', 
        'prophet',
        'tensorflow',
        'torch',
        'plotly',
        'streamlit'
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    
    print("All packages installed successfully!")

def clone_repository():
    """Clone the repository."""
    import os
    import subprocess
    
    if not os.path.exists('energy_price_predictor'):
        print("Cloning repository...")
        subprocess.check_call(['git', 'clone', 'https://github.com/tommasomalaguti/energy_price_predictor.git'])
        print("Repository cloned successfully!")
    else:
        print("Repository already exists!")
    
    # Change to project directory
    os.chdir('energy_price_predictor')
    print(f"Changed to directory: {os.getcwd()}")

if __name__ == "__main__":
    install_packages()
    clone_repository()
    print("\nSetup complete! You can now run the forecasting models.")
