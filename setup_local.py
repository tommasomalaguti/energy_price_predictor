#!/usr/bin/env python3
"""
Local setup script for electricity price forecasting project.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_local.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def test_imports():
    """Test if all required packages can be imported."""
    print("🧪 Testing imports...")
    
    required_packages = [
        'pandas', 'numpy', 'requests', 'bs4', 'matplotlib', 
        'seaborn', 'plotly', 'sklearn', 'statsmodels'
    ]
    
    optional_packages = [
        'xgboost', 'lightgbm', 'prophet', 'tensorflow', 'torch'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            failed_imports.append(package)
    
    print("\nOptional packages:")
    for package in optional_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ⚠️  {package} (optional)")
    
    if failed_imports:
        print(f"\n❌ Failed to import required packages: {failed_imports}")
        return False
    else:
        print("\n✅ All required packages imported successfully!")
        return True

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = [
        'data/raw',
        'data/processed', 
        'models/saved',
        'results',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✅ {directory}")

def main():
    """Main setup function."""
    print("🚀 Setting up local development environment...")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed at requirements installation")
        return False
    
    # Test imports
    if not test_imports():
        print("❌ Setup failed at import testing")
        return False
    
    # Create directories
    create_directories()
    
    print("\n🎉 Local setup complete!")
    print("\nNext steps:")
    print("1. Run: python debug_entsoe_api.py")
    print("2. Check the results and debug any API issues")
    print("3. Once working, run the full forecasting workflow")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
