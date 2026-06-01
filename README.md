# E-Commerce End-to-End Master Segmentation Engine

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![ML Framework](https://img.shields.io/badge/machine%20learning-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![BI Analytics](https://img.shields.io/badge/analytics-Tableau-orange.svg)](https://public.tableau.com/)
[![UI App](https://img.shields.io/badge/interface-Streamlit-red.svg)](https://streamlit.io/)

An enterprise-grade customer analytics platform that orchestrates raw marketplace transaction parsing, evaluates text sentiment profiles via NLP lexicons, handles multidimensional feature scaling, executes automated K-Means clustering optimizations, and provisions interactive live monitoring layers.

## 📐 System Architecture
The system execution workflow is completely decoupled into dedicated lifecycle operations for production scalability:

1. **Workspace Environment Provisioner:** Dynamically configures directory paths and configuration boundaries (`config.yaml`).
2. **Text Processing NLP Core:** Integrates `TextBlob` semantic modeling to calculate customer review sentiment polarity.
3. **Variance Scaling & Transformation:** Leverages `StandardScaler` arrays to balance multi-unit feature spaces (`Recency_Days`, `Frequency`, `Monetary`, `Average_Sentiment`) smoothly prior to vector processing.
4. **Algorithmic K-Means Optimizer:** Fits and scales cluster assignments automatically across dynamic client profiles.
5. **UI Application Layer:** Provisions a functional `Streamlit` configuration template (`app.py`) for live interactive database exploration.

---

## 📁 Repository Directory Layout
```text
├── config.yaml                     # Machine learning pipeline hyper-parameters
├── requirements.txt                # Complete environment library bindings
├── app.py                          # Streamlit UI front-end app server script
├── README.md                       # Comprehensive system documentation
├── src/                            # Production pipeline module operations
│   ├── preprocessing.py            # Extraction and aggregation transformations
│   ├── sentiment.py                # NLP text evaluation engine
│   ├── clustering.py               # Auto-tuning K-Means iteration core model
│   ├── visualization.py            # Pipeline diagnostic plotting mechanisms
│   └── pipeline.py                 # Core orchestration execution manager
├── tests/                          # Quality assurance system test suite
│   ├── test_preprocessing.py       # Preprocessing unit checks
│   └── test_clustering.py          # Machine learning stability tests
├── outputs/                        # Processed target dataset outputs
│   └── csv/
│       └── segment_output.csv
└── visualizations/                 # Enterprise business intelligence deliverables
    ├── Customer_Segmentation_Dashboard.twbx  # Packaged Tableau workbook asset
    └── sentiment_chart.png         # Executive dashboard reporting preview
