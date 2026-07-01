# E-Commerce Customer Segmentation Intelligence Engine
## Behavioral Segmentation | RFM Analytics, NLP Sentiment & Tableau

---

## Project Overview

This project is an end-to-end e-commerce customer intelligence solution designed to turn raw marketplace review and purchase-value signals into actionable behavioral segments.

The pipeline cleans public e-commerce records, engineers RFM-style features, extracts sentiment from review text, selects an appropriate number of K-Means clusters using silhouette analysis, and translates technical clusters into business-ready customer cohorts.

The final outputs support two decision-making layers:

- **Tableau executive dashboard** for segment performance, revenue contribution, sentiment patterns, and customer-level exploration.
- **Streamlit analytics app** for interactive KPI monitoring and segment-level data exploration.

The project is built with a modular, configuration-driven Python architecture so the same workflow can be extended to a production customer-data source.

---

## Business Problem

E-commerce teams collect large volumes of transactional and review data, but often struggle to answer:

- Which customer groups create the most value?
- Which groups show dissatisfaction signals and may need intervention?
- How should marketing, retention, and loyalty efforts be prioritized?
- Which segments should receive premium experiences, win-back offers, or personalized campaigns?
- How can customer behavior and review sentiment be analyzed together rather than separately?

This project addresses those questions by combining RFM-style behavioral analytics with NLP-based sentiment scoring and unsupervised machine learning.

---

## Project Objectives

1. Build a reusable e-commerce data-preprocessing workflow.
2. Create customer-level RFM-style features from recent e-commerce activity.
3. Extract customer sentiment from aggregated review text.
4. Identify natural behavioral clusters using K-Means.
5. Convert technical clusters into clear, business-friendly strategic segments.
6. Deliver dashboard and application layers that support marketing and retention decisions.

---

## Tech Stack

| Layer | Tools Used |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Configuration | YAML |
| Feature Engineering | Recency, Frequency, Monetary, Sentiment |
| NLP | TextBlob |
| Machine Learning | Scikit-learn, K-Means, StandardScaler, PCA, Silhouette Score |
| Visualization | Matplotlib, Seaborn, Tableau |
| Interactive App | Streamlit |
| Testing | Pytest |
| Version Control | GitHub |

---

## Repository Structure

```text
ECommerce-Customer-Segmentation-Engine/
│
├── dashboard/
│   ├── Customer_Segmentation_Dashboard.twb
│   └── E-Commerce Customer Segmentation Dashboard.png
│
├── notebooks/
│   └── ECommerce_End_To_End_Master_Engine (1).ipynb
│
├── outputs/
│   ├── csv/
│   │   └── segment_output (1).csv
│   ├── reports/                        # Generated when the pipeline runs
│   └── images/                         # Generated when the pipeline runs
│
├── src/
│   ├── preprocessing.py                # Ingestion, cleaning, RFM-style aggregation
│   ├── sentiment.py                    # TextBlob sentiment scoring
│   ├── clustering.py                   # Scaling, K-Means, PCA, silhouette selection
│   ├── visualization.py                # Analytical visual generation
│   └── pipeline.py                     # End-to-end orchestration
│
├── tests/
│   ├── test_clustering.py
│   └── test_preprocessing.py
│
├── app.py                              # Streamlit application
├── config.yaml                         # Data source and pipeline configuration
├── requirements.txt
└── README.md
```

---

## Dataset and Analytical Scope

The workflow is configured to ingest a public Amazon e-commerce sample. The preprocessing layer cleans the price field, parses customer-review information, limits analysis to a rolling 12-month window, and creates an entity-level analytical table.

### Core Input Signals

| Source Field | Analytical Use |
|---|---|
| `uniq_id` | Entity identifier used to build the analytical profile |
| `price` | Monetary-value proxy |
| `customer_reviews` | Review date, rating, and review text |
| Parsed review date | Recency calculation |
| Aggregated review text | NLP sentiment scoring |

### Derived Analytical Features

| Feature | Description |
|---|---|
| `Recency_Days` | Days since the entity's most recent observed review/activity |
| `Frequency` | Count of observed records |
| `Monetary` | Total price/value proxy |
| `Average_Rating` | Average extracted star rating |
| `Average_Sentiment` | TextBlob polarity score from aggregated review text |
| `Sentiment_Label` | Positive, Neutral, or Negative |
| `Cluster_ID` | K-Means cluster assignment |
| `Strategic_Segment` | Business-friendly cohort label |

> **Data-design note:** This repository is a portfolio proof of concept built on a public sample. In a production deployment, replace the configured source with order-level data containing a true `customer_id`, transaction date, order value, and customer feedback. That change enables fully validated customer-level RFM segmentation.

---

## Data Architecture

```text
Public E-Commerce Source
        ↓
Data Ingestion and Cleaning
        ↓
RFM-Style Feature Engineering
        ↓
Review Aggregation and NLP Sentiment Scoring
        ↓
StandardScaler Feature Normalization
        ↓
K-Means Model Selection via Silhouette Score
        ↓
PCA Coordinates for Cluster Visualization
        ↓
Business-Friendly Strategic Segment Labels
        ↓
CSV Outputs, Tableau Dashboard and Streamlit App
```

---

## Segmentation Methodology

### 1. Data Cleaning and Preparation

The preprocessing module:

- Downloads the source dataset defined in `config.yaml`.
- Cleans price values into numerical fields.
- Parses rating, date, and review-text signals from review records.
- Fills missing numerical values using robust defaults.
- Filters the data to the most recent configured rolling window.
- Aggregates records into entity-level analytical profiles.

### 2. RFM-Style Feature Engineering

The solution uses four standardized clustering inputs:

- **Recency:** time since recent observed activity.
- **Frequency:** count of observed activity records.
- **Monetary:** cumulative value proxy.
- **Average Sentiment:** satisfaction or dissatisfaction signal from text feedback.

This combination captures both behavioral value and customer experience.

### 3. NLP Sentiment Analysis

TextBlob calculates a polarity score from aggregated review text:

- **Positive:** sentiment score above `0.05`
- **Neutral:** sentiment score between `-0.05` and `0.05`
- **Negative:** sentiment score below `-0.05`

Sentiment becomes an additional feature in the clustering workflow, allowing the model to distinguish high-value satisfied cohorts from dissatisfied or at-risk cohorts.

### 4. K-Means Cluster Selection

The engine evaluates candidate cluster counts from **K = 2** through **K = 6**.

For each candidate:

1. Input features are standardized using `StandardScaler`.
2. A K-Means model is trained with a reproducible random state.
3. The silhouette score is calculated.
4. The cluster count with the strongest silhouette score is selected.

PCA reduces standardized features to two dimensions for cluster visualization.

### 5. Strategic Segment Mapping

Raw K-Means cluster IDs are translated into business-readable cohorts:

| Segment | Identification Logic | Recommended Business Focus |
|---|---|---|
| **Premium Champions** | Highest monetary-value cluster after excluding the lowest-sentiment cluster | Loyalty, early-access offers, premium retention |
| **At-Risk Detractors** | Lowest average-sentiment cluster | Service recovery, feedback follow-up, targeted win-back actions |
| **Value Seekers** | Remaining behavioral cluster(s) | Personalized promotions, cross-sell, value-led campaigns |

> Strategic labels are applied after unsupervised clustering to make the model output actionable for business users.

---

## Pipeline Outputs

Running the pipeline produces:

| Output | Description |
|---|---|
| `outputs/csv/segment_output.csv` | Entity-level analytical dataset with RFM-style fields, sentiment, cluster IDs, PCA coordinates, and strategic segments |
| `outputs/reports/cluster_summary.csv` | Average Recency, Frequency, Monetary, and Sentiment by strategic segment |
| `outputs/images/` | Generated analytical visuals, including clustering diagnostics |
| Tableau workbook | Executive-facing segmentation dashboard |
| Streamlit app | Interactive KPI, chart, and data-explorer interface |

---

## Tableau Dashboard

![E-Commerce Customer Segmentation Dashboard](dashboard/E-Commerce%20Customer%20Segmentation%20Dashboard.png)

The Tableau dashboard translates the analytical output into an executive customer-intelligence view.

### Dashboard Capabilities

- Customer-segment distribution
- Revenue/value contribution by strategic segment
- RFM behavior comparison across cohorts
- Sentiment-performance analysis
- Customer-level drill-down capability
- Business insight area for marketing and retention decisions

Open the Tableau workbook here:

[`dashboard/Customer_Segmentation_Dashboard.twb`](dashboard/Customer_Segmentation_Dashboard.twb)

---

## Streamlit Analytics Application

The Streamlit interface provides an additional interactive exploration layer.

### Available Views

- Total entities evaluated
- Total marketplace value
- Average activity frequency
- Average sentiment score
- Segment-level financial contribution chart
- Sentiment distribution analysis
- Filterable strategic-segment data explorer

Run it locally after the pipeline has created the output file.

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/sharath-redddy/ECommerce-Customer-Segmentation-Engine.git
cd ECommerce-Customer-Segmentation-Engine
```

### 2. Create and Activate a Virtual Environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the End-to-End Pipeline

```bash
python src/pipeline.py
```

This downloads the configured source data, creates the RFM-style and sentiment features, selects the optimal K-Means configuration, assigns strategic segments, generates outputs, and writes the analytical files.

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

### 6. Run Tests

```bash
pip install pytest
pytest -q
```

---

## Business Recommendations

Based on the segment design, e-commerce teams can act on the results as follows:

1. **Protect Premium Champions** with loyalty rewards, exclusive access, proactive service, and personalized premium offers.
2. **Recover At-Risk Detractors** using review follow-ups, service interventions, customer-support escalation, and targeted win-back campaigns.
3. **Grow Value Seekers** through bundles, price-sensitive promotions, product recommendations, and cross-sell journeys.
4. **Use sentiment with RFM** so that high-value but dissatisfied entities do not get overlooked.
5. **Refresh segments regularly** to detect movement from healthy to dissatisfied behavior before value is lost.
6. **Connect outputs to campaign systems** so segment-specific actions can be measured and optimized.

---

## Business Impact

This solution demonstrates how an e-commerce organization can:

- Move from generic marketing to cohort-based customer engagement.
- Combine behavioral value and sentiment into one decision framework.
- Identify high-value and potentially dissatisfied cohorts early.
- Prioritize retention and promotion budgets with data.
- Give business users self-service access through Tableau and Streamlit.
- Reuse a modular Python pipeline when the data source or business rules change.

---

## Future Enhancements

- Replace the public proof-of-concept source with real order-level customer data.
- Add customer lifetime value and predicted churn-risk features.
- Track segment migration over time.
- Use topic modeling or transformer-based sentiment classification.
- Add campaign-response and uplift analysis.
- Automate pipeline execution with a scheduler or CI/CD workflow.
- Publish the Streamlit application to a cloud platform.

---

## Author

**Sharath Reddy**  
Data Analytics | Machine Learning | Business Intelligence
