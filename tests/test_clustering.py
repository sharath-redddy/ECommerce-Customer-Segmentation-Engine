import pytest
import pandas as pd
import numpy as np
from src.clustering import CustomerSegmenter

@pytest.fixture
def mock_config():
    return {
        'pipeline': {
            'random_state': 42,
            'max_k': 4
        },
        'features': {
            'clustering_cols': ['Recency_Days', 'Frequency', 'Monetary', 'Average_Sentiment']
        }
    }

@pytest.fixture
def mock_customer_data():
    # Build synthetic sample frame
    np.random.seed(42)
    return pd.DataFrame({
        'User_ID': [f'User_{i}' for i in range(20)],
        'Recency_Days': np.random.randint(1, 100, 20),
        'Frequency': np.random.randint(1, 15, 20),
        'Monetary': np.random.uniform(10.0, 500.0, 20),
        'Average_Sentiment': np.random.uniform(-1.0, 1.0, 20)
    })

def test_kmeans_and_pca_dimensions(mock_config, mock_customer_data):
    segmenter = CustomerSegmenter(mock_config)
    
    # Validate initialization parameter limits
    chosen_k = segmenter.find_best_k(mock_customer_data)
    assert 2 <= chosen_k <= mock_config['pipeline']['max_k']
    
    # Validate processed pipeline data transformations
    output_df = segmenter.fit_clusters(mock_customer_data)
    assert 'Cluster_ID' in output_df.columns
    assert 'Strategic_Segment' in output_df.columns
    assert 'PCA_1' in output_df.columns
    assert 'PCA_2' in output_df.columns
