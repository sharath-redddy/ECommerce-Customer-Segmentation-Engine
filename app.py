import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configure professional enterprise theme layout
st.set_page_config(
    page_title="E-Commerce Master Segmentation Engine",
    page_icon="📊",
    layout="wide"
)

st.title("📊 E-Commerce Customer Segmentation & Sentiment Hub")
st.markdown("---")

# Target production output data pathway
DATA_PATH = "outputs/csv/segment_output.csv"

if os.path.exists(DATA_PATH):
    # Load processed target dataset pipelines
    df = pd.read_csv(DATA_PATH)
    
    # --- ENTERPRISE KPI HIGHLIGHT TRACKING METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers Evaluated", f"{df['User_ID'].nunique():,}")
    with col2:
        st.metric("Total Marketplace Revenue", f"₹{df['Monetary'].sum():,.2f}")
    with col3:
        st.metric("Average Customer Frequency", f"{df['Frequency'].mean():.1f} Orders")
    with col4:
        st.metric("Average Sentiment Score", f"{df['Average_Sentiment'].mean():+.2f}")
        
    st.markdown("---")
    
    # --- MULTI-DIMENSIONAL SPLIT SCREEN ANALYSIS LAYOUT ---
    left_chart, right_chart = st.columns(2)
    
    with left_chart:
        st.subheader("📈 Cohort Financial Contribution Metrics")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=df, x="Strategic_Segment", y="Monetary", estimator=sum, errorbar=None, palette="Set2", ax=ax)
        plt.xticks(rotation=15)
        plt.ylabel("Total Financial Volume (₹)")
        st.pyplot(fig)
        
    with right_chart:
        st.subheader("🎯 Sentiment Distribution Profile Space")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="Sentiment_Label", y="Average_Sentiment", palette="Pastel1", ax=ax)
        st.pyplot(fig)
        
    st.markdown("---")
    
    # --- LIVE DATA EXPLORER FILTER SECTIONS ---
    st.subheader("🔍 Production Segment Data Explorer Registry")
    selected_segment = st.selectbox("Filter System Register By Strategic Cohort:", df['Strategic_Segment'].unique())
    filtered_df = df[df['Strategic_Segment'] == selected_segment]
    
    st.dataframe(
        filtered_df[['User_ID', 'Recency_Days', 'Frequency', 'Monetary', 'Average_Sentiment', 'Sentiment_Label']], 
        use_container_width=True
    )
    
else:
    # Fail-safe warning layout for clean system deployments
    st.warning("⚠️ No active dataset processing history found. Run the core system pipeline first (`python src/pipeline.py`) to generate the required analytical outputs.")
