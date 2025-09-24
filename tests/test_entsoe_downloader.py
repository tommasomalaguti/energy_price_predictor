"""
Tests for ENTSO-E downloader module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock
from src.data.entsoe_downloader import ENTSOEDownloader


class TestENTSOEDownloader:
    """Test cases for ENTSOEDownloader class."""
    
    def test_init_with_token(self):
        """Test initialization with API token."""
        token = "test_token_123"
        downloader = ENTSOEDownloader(api_token=token)
        assert downloader.api_token == token
    
    def test_init_without_token(self):
        """Test initialization without API token."""
        with patch.dict('os.environ', {'ENTSOE_API_TOKEN': 'env_token'}):
            downloader = ENTSOEDownloader()
            assert downloader.api_token == 'env_token'
    
    def test_init_no_token_available(self):
        """Test initialization when no token is available."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError):
                ENTSOEDownloader()
    
    @patch('requests.Session.get')
    def test_test_api_connection_success(self, mock_get):
        """Test successful API connection test."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<?xml version='1.0'?><test>success</test>"
        mock_get.return_value = mock_response
        
        downloader = ENTSOEDownloader(api_token="test_token")
        result = downloader.test_api_connection()
        
        assert result is True
        mock_get.assert_called_once()
    
    @patch('requests.Session.get')
    def test_test_api_connection_unauthorized(self, mock_get):
        """Test API connection test with unauthorized response."""
        # Mock unauthorized response
        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response
        
        downloader = ENTSOEDownloader(api_token="invalid_token")
        result = downloader.test_api_connection()
        
        assert result is False
    
    @patch('requests.Session.get')
    def test_download_chunk_success(self, mock_get, mock_entsoe_response):
        """Test successful data download."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = mock_entsoe_response
        mock_response.headers = {'content-type': 'application/xml'}
        mock_get.return_value = mock_response
        
        downloader = ENTSOEDownloader(api_token="test_token")
        result = downloader._download_chunk("DE", "2023-01-01", "2023-01-01", "day_ahead")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'price' in result.columns
        assert 'datetime' in result.columns
    
    @patch('requests.Session.get')
    def test_download_chunk_unauthorized(self, mock_get):
        """Test data download with unauthorized response."""
        # Mock unauthorized response
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_response.headers = {'content-type': 'application/xml'}
        mock_get.return_value = mock_response
        
        downloader = ENTSOEDownloader(api_token="invalid_token")
        result = downloader._download_chunk("DE", "2023-01-01", "2023-01-01", "day_ahead")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    @patch('requests.Session.get')
    def test_download_chunk_acknowledgement_document(self, mock_get):
        """Test data download with acknowledgement document (no data)."""
        # Mock acknowledgement document response
        ack_response = """<?xml version="1.0" encoding="UTF-8"?>
        <Acknowledgement_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-1:acknowledgementdocument:7:0">
            <mRID>test123</mRID>
            <createdDateTime>2023-01-01T00:00:00Z</createdDateTime>
        </Acknowledgement_MarketDocument>"""
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = ack_response
        mock_get.return_value = mock_response
        
        downloader = ENTSOEDownloader(api_token="test_token")
        result = downloader._download_chunk("DE", "2023-01-01", "2023-01-01", "day_ahead")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_parse_xml_response_success(self, mock_entsoe_response):
        """Test successful XML parsing."""
        downloader = ENTSOEDownloader(api_token="test_token")
        result = downloader._parse_xml_response(mock_entsoe_response, "day_ahead")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'price' in result.columns
        assert 'datetime' in result.columns
        assert result['price'].dtype in [np.float64, np.int64]
    
    def test_parse_xml_response_invalid_xml(self):
        """Test XML parsing with invalid XML."""
        downloader = ENTSOEDownloader(api_token="test_token")
        result = downloader._parse_xml_response("invalid xml", "day_ahead")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_parse_xml_response_empty_response(self):
        """Test XML parsing with empty response."""
        downloader = ENTSOEDownloader(api_token="test_token")
        result = downloader._parse_xml_response("", "day_ahead")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_get_available_countries(self):
        """Test getting available countries."""
        downloader = ENTSOEDownloader(api_token="test_token")
        countries = downloader.get_available_countries()
        
        assert isinstance(countries, dict)
        assert len(countries) > 0
        # Check for some expected countries
        assert 'DE' in countries
        assert 'FR' in countries
        assert 'IT' in countries
    
    @patch('requests.Session.get')
    def test_download_price_data_success(self, mock_get, mock_entsoe_response):
        """Test successful price data download."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = mock_entsoe_response
        mock_get.return_value = mock_response
        
        downloader = ENTSOEDownloader(api_token="test_token")
        result = downloader.download_price_data("DE", "20230101", "20230101")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'price' in result.columns
        assert 'datetime' in result.columns
    
    @patch('requests.Session.get')
    def test_download_price_data_with_save(self, mock_get, mock_entsoe_response, tmp_path):
        """Test price data download with saving to file."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = mock_entsoe_response
        mock_get.return_value = mock_response
        
        downloader = ENTSOEDownloader(api_token="test_token")
        save_path = tmp_path / "test_data.csv"
        result = downloader.download_price_data("DE", "20230101", "20230101", str(save_path))
        
        assert isinstance(result, pd.DataFrame)
        assert save_path.exists()
        
        # Verify saved data
        saved_data = pd.read_csv(save_path)
        assert len(saved_data) > 0
        assert 'price' in saved_data.columns
    
    def test_domain_code_generation(self):
        """Test domain code generation for different countries."""
        downloader = ENTSOEDownloader(api_token="test_token")
        
        # Test different country codes
        test_cases = [
            ("DE", "10Y1001A1001A63L"),
            ("FR", "10YFR-RTE------C"),
            ("IT", "10YIT----------"),
            ("ES", "10YES-REE------0"),
            ("NL", "10YNL----------L")
        ]
        
        for country, expected_domain in test_cases:
            domain = downloader._get_domain_code(country)
            assert domain == expected_domain
    
    def test_invalid_country_code(self):
        """Test handling of invalid country code."""
        downloader = ENTSOEDownloader(api_token="test_token")
        
        with pytest.raises(ValueError):
            downloader._get_domain_code("INVALID")
    
    @patch('requests.Session.get')
    def test_network_error_handling(self, mock_get):
        """Test handling of network errors."""
        # Mock network error
        mock_get.side_effect = Exception("Network error")
        
        downloader = ENTSOEDownloader(api_token="test_token")
        result = downloader._download_chunk("DE", "2023-01-01", "2023-01-01", "day_ahead")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
