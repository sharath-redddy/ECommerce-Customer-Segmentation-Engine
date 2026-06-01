import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger("Pipeline")

class CustomerSegmenter:
    def __init__(self, config: dict):
        self.config = config
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.best_k = 3
        self.elbow_scores = []
        self.silhouette_scores = {}

    def find_best_k(self, df: pd.DataFrame) -> int:
        features = self.config['features']['clustering_cols']
        scaled_data = self.scaler.fit_transform(df[features])
        max_k = self.config['pipeline']['max_k']

        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=self.config['pipeline']['random_state'], n_init=10)
            labels = km.fit_predict(scaled_data)
            self.elbow_scores.append(km.inertia_)
            self.silhouette_scores[k] = silhouette_score(scaled_data, labels)

        self.best_k = max(self.silhouette_scores, key=self.silhouette_scores.get)
        logger.info(f"Optimal cluster K selected: K={self.best_k}")
        return self.best_k

    def fit_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self.config['features']['clustering_cols']
        scaled_features = self.scaler.fit_transform(df[features])

        km = KMeans(n_clusters=self.best_k, random_state=self.config['pipeline']['random_state'], n_init=10)
        df['Cluster_ID'] = km.fit_predict(scaled_features)

        pca_coords = self.pca.fit_transform(scaled_features)
        df['PCA_1'] = pca_coords[:, 0]
        df['PCA_2'] = pca_coords[:, 1]

        profiles = df.groupby('Cluster_ID')[features].mean()
        detractor_id = profiles['Average_Sentiment'].idxmin()
        premium_id = profiles.drop(index=detractor_id)['Monetary'].idxmax()

        label_map = {}
        for c_id in profiles.index:
            if c_id == detractor_id:
                label_map[c_id] = "At-Risk Detractors"
            elif c_id == premium_id:
                label_map[c_id] = "Premium Champions"
            else:
                label_map[c_id] = "Value Seekers"

        df['Strategic_Segment'] = df['Cluster_ID'].map(label_map)
        return df
