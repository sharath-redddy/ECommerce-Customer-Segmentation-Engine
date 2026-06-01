import pytest
import pandas as pd
import numpy as np
from src.preprocessing import DataPreprocessor

@pytest.fixture
def mock_config():
    return {
        'data': {
            'stream_url': '',
            'raw_path': 'data/raw/test.csv'
        },
        'pipeline': {
            'rolling_months': 12
        }
    }

def test_currency_and_price_cleaning(mock_config):
    processor = DataPreprocessor(mock_config)
    
    # Assert successful handling of varied currency syntax symbols
    assert processor._clean_price("£15.99") == 15.99
    assert processor._clean_price("$45.00 - $50.00") == 45.00
    assert processor._clean_price(" 120.50 ") == 120.50
    assert np.isnan(processor._clean_price("Out of Stock/Free"))

def test_star_rating_extraction(mock_config):
    processor = DataPreprocessor(mock_config)
    
    # Assert extraction of numerical values from raw text formats
    assert processor._extract_rating("4.2 out of 5 stars") == 4.2
    assert processor._extract_rating("5.0 out of 5 stars // review details") == 5.0
    assert np.isnan(processor._extract_rating("No ratings present yet"))
