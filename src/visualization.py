import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class PipelineVisualizer:
    def __init__(self, config: dict):
        self.config = config
        self.out_dir = config['data']['output_img_dir']
        os.makedirs(self.out_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def generate_plots(self, df: pd.DataFrame, segmenter_obj):
        # 1. Elbow and Silhouette diagnostic curves
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(range(2, len(segmenter_obj.elbow_scores)+2), segmenter_obj.elbow_scores, 'b-', marker='o')
        ax1.set_xlabel('Number of Clusters (K)')
        ax1.set_ylabel('Inertia (Elbow)', color='b')
        ax2 = ax1.twinx()
        ax2.plot(list(segmenter_obj.silhouette_scores.keys()), list(segmenter_obj.silhouette_scores.values()), 'g-', marker='s')
        ax2.set_ylabel('Silhouette Score', color='g')
        plt.title('Clustering Optimization Diagnostics')
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "model_optimization_metrics.png"), dpi=300)
        plt.close()

        # 2. PCA Scatter space coordinates projection
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x='PCA_1', y='PCA_2', hue='Strategic_Segment', palette='Set2', alpha=0.8)
        plt.title('Customer Segments: 2D PCA Space Projection')
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "pca_cluster_projection.png"), dpi=300)
        plt.close()

        # 3. Strategic Heatmap Matrix profile summary
        features = self.config['features']['clustering_cols']
        matrix = df.groupby('Strategic_Segment')[features].mean()
        plt.figure(figsize=(8, 4))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=1)
        plt.title('Customer Strategic Cohort Behavioral Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "strategic_cohort_heatmap.png"), dpi=300)
        plt.close()
