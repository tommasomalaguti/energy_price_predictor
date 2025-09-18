#!/usr/bin/env python3
"""
Setup script for electricity price forecasting project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="energy-price-predictor",
    version="0.1.0",
    author="Energy Price Forecasting Team",
    author_email="team@energyforecast.com",
    description="A comprehensive toolkit for forecasting day-ahead electricity market prices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/energy_price_predictor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
        ],
        "deep": [
            "tensorflow>=2.10.0",
            "torch>=1.12.0",
        ],
        "dashboard": [
            "streamlit>=1.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "energy-forecast=example_workflow:main",
        ],
    },
)
