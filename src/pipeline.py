import yaml
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Pipeline")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import DataPreprocessor
from sentiment import SentimentAnalyzer
from clustering import CustomerSegmenter
from visualization import PipelineVisualizer

def run_master_pipeline(config_path: str = "config.yaml"):
    logger.info("Starting customer segmentation pipeline...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    preprocessor = DataPreprocessor(config)
    preprocessor.load_data()
    raw_clean = preprocessor.clean_and_filter()
    customer_profiles = preprocessor.aggregate_customers(raw_clean)

    analyzer = SentimentAnalyzer()
    customer_profiles = analyzer.calculate_sentiment(customer_profiles)

    segmenter = CustomerSegmenter(config)
    segmenter.find_best_k(customer_profiles)
    final_segmented_data = segmenter.fit_clusters(customer_profiles)

    visualizer = PipelineVisualizer(config)
    visualizer.generate_plots(final_segmented_data, segmenter)

    # Save final analytical tracking spreadsheet
    final_segmented_data.to_csv(config['data']['output_csv'], index=False)

    # Save cluster summary report deliverable for stakeholders
    os.makedirs(os.path.dirname(config['data']['output_summary']), exist_ok=True)
    cluster_summary = (
        final_segmented_data.groupby("Strategic_Segment")
        .agg({
            "Recency_Days": "mean",
            "Frequency": "mean",
            "Monetary": "mean",
            "Average_Sentiment": "mean"
        })
    )
    cluster_summary.to_csv(config['data']['output_summary'])
    logger.info(f"Saved executive cluster summary report to: {config['data']['output_summary']}")

    # Calculate exact contribution values for the portfolio README metadata section
    total_spend = final_segmented_data['Monetary'].sum()
    champions = final_segmented_data[final_segmented_data['Strategic_Segment'] == "Premium Champions"]
    premium_spend_pct = (champions['Monetary'].sum() / total_spend * 100) if total_spend > 0 else 0.0

    readme_content = f"""# Customer Segmentation Pipeline via RFM & NLP Sentiment

An end-to-end machine learning pipeline that aggregates raw marketplace transactions into customer profiles, extracts NLP sentiment metrics from text feedback, and groups behaviors using K-Means clustering.

## Results

- **Optimal K selected automatically**: {segmenter.best_k}
- **Best Silhouette Score**: {max(segmenter.silhouette_scores.values()):.2f}
- **Identified Segments**: {final_segmented_data['Strategic_Segment'].nunique()} Unique Customer Behavioral Cohorts
- **Premium Champions Value Metric**: Premium customers accounted for {premium_spend_pct:.1f}% of total marketplace spend.
"""
    with open("README.md", "w") as f:
        f.write(readme_content)

    logger.info("Pipeline completed successfully.")

if __name__ == '__main__':
    run_master_pipeline()
