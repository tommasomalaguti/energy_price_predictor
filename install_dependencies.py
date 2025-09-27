#!/usr/bin/env python3
"""
Dependency installer for Electricity Price Forecasting Dashboard.

This script helps install the required dependencies and provides
instructions for fixing common issues.
"""

import subprocess
import sys
import platform
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Installing {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def install_openmp_macos():
    """Install OpenMP runtime for macOS."""
    print("\n=== Installing OpenMP Runtime for macOS ===")
    
    # Check if Homebrew is available
    try:
        subprocess.run(["brew", "--version"], check=True, capture_output=True)
        print("✓ Homebrew is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Homebrew not found. Please install Homebrew first:")
        print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        return False
    
    # Install libomp
    success = run_command("brew install libomp", "OpenMP runtime (libomp)")
    if success:
        print("✓ OpenMP runtime installed successfully")
        return True
    else:
        print("✗ Failed to install OpenMP runtime")
        return False

def check_system():
    """Check the operating system and provide system-specific instructions."""
    system = platform.system()
    print(f"Detected system: {system}")
    
    if system == "Darwin":  # macOS
        print("\n=== macOS Setup ===")
        # Try to install OpenMP automatically
        openmp_success = install_openmp_macos()
        if not openmp_success:
            print("\nManual installation required:")
            print("1. Install Homebrew if not already installed:")
            print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
            print("2. Install OpenMP runtime:")
            print("   brew install libomp")
        
    elif system == "Linux":
        print("\n=== Linux Instructions ===")
        print("1. Install OpenMP development libraries:")
        print("   sudo apt-get install libomp-dev  # Ubuntu/Debian")
        print("   sudo yum install libgomp-devel    # CentOS/RHEL")
        print("\n2. Install Python dependencies:")
        print("   pip install -r requirements-dashboard.txt")
        
    elif system == "Windows":
        print("\n=== Windows Instructions ===")
        print("1. Install Microsoft Visual C++ Redistributable")
        print("2. Install Python dependencies:")
        print("   pip install -r requirements-dashboard.txt")
    
    return system

def install_python_dependencies():
    """Install Python dependencies."""
    print("\n=== Installing Python Dependencies ===")
    
    # Install core dependencies first
    core_deps = [
        "streamlit>=1.28.0",
        "plotly>=5.15.0", 
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "requests>=2.28.0",
        "beautifulsoup4>=4.11.0"
    ]
    
    for dep in core_deps:
        success = run_command(f"pip install '{dep}'", dep)
        if not success:
            print(f"Warning: Failed to install {dep}")
    
    # Try to install optional dependencies
    optional_deps = [
        "xgboost>=1.7.0",
        "lightgbm>=3.3.0",
        "prophet>=1.1.0"
    ]
    
    print("\n=== Installing Optional Dependencies ===")
    for dep in optional_deps:
        success = run_command(f"pip install '{dep}'", dep)
        if not success:
            print(f"Note: {dep} failed to install - this is optional")

def test_imports():
    """Test if critical imports work."""
    print("\n=== Testing Imports ===")
    
    test_imports = [
        ("streamlit", "Streamlit"),
        ("plotly", "Plotly"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
        ("requests", "Requests"),
        ("bs4", "BeautifulSoup4")
    ]
    
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✓ {name} - OK")
        except ImportError:
            print(f"✗ {name} - Missing")
    
    # Test optional imports
    optional_imports = [
        ("xgboost", "XGBoost"),
        ("lightgbm", "LightGBM"),
        ("prophet", "Prophet")
    ]
    
    print("\n=== Optional Dependencies ===")
    for module, name in optional_imports:
        try:
            __import__(module)
            print(f"✓ {name} - Available")
        except ImportError:
            print(f"WARNING {name} - Not available (optional)")

def main():
    """Main installation script."""
    print("Electricity Price Forecasting - Dependency Installer")
    print("=" * 50)
    
    # Check system
    system = check_system()
    
    # Install Python dependencies
    install_python_dependencies()
    
    # Test imports
    test_imports()
    
    print("\n=== Installation Complete ===")
    print("You can now run the dashboard with:")
    print("  streamlit run dashboard.py")
    
    if system == "Darwin":
        print("\nIf you encounter XGBoost issues, run:")
        print("  brew install libomp")
        print("Then restart the dashboard.")

if __name__ == "__main__":
    main()
