# E-Commerce Customer Segmentation Engine
⭐ End-to-end Data Analytics Project | Machine Learning | NLP | Tableau | Streamlit

An end-to-end customer analytics and segmentation platform built using Python, Machine Learning, NLP, Tableau, and Streamlit.

This project analyzes e-commerce customer behavior using RFM (Recency, Frequency, Monetary) metrics and customer review sentiment to identify meaningful customer segments. By combining sentiment analysis with K-Means clustering, the system helps businesses understand customer value, detect at-risk users, and design targeted retention and marketing strategies.

The project follows a modular and production-oriented architecture with separate components for data preprocessing, sentiment analysis, clustering, visualization, testing, and dashboard reporting.

## Business Problem

E-commerce companies often collect large volumes of transaction and review data but struggle to convert that information into actionable customer insights.

The goal of this project is to:

* Identify high-value customer groups
* Detect customers at risk of churn
* Understand customer sentiment patterns
* Support data-driven marketing and retention strategies
* Visualize customer behavior through an interactive business dashboard

## Solution Overview

The pipeline processes raw customer transaction and review data through multiple analytical stages:

1. Data Cleaning and Preprocessing

   * Handles missing values and inconsistent records
   * Generates customer-level analytical features

2. RFM Feature Engineering

   * Recency: Days since last purchase
   * Frequency: Number of purchases
   * Monetary: Total customer spending

3. Sentiment Analysis

   * Uses TextBlob NLP techniques
   * Extracts sentiment polarity scores from customer reviews

4. Feature Scaling

   * Standardizes numerical variables using StandardScaler

5. Customer Segmentation

   * Applies K-Means clustering
   * Groups customers into strategic behavioral segments

6. Business Intelligence Layer

   * Interactive Tableau dashboard
   * Customer performance monitoring
   * Segment-level business insights

## Key Features

* Modular production-style Python architecture
* Automated data preprocessing pipeline
* RFM customer analysis
* NLP-based sentiment scoring
* K-Means customer segmentation
* Tableau business intelligence dashboard
* Streamlit application interface
* Unit testing for core pipeline modules
* Configuration-driven workflow using YAML

## Technology Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* TextBlob
* Matplotlib
* Seaborn
* Tableau
* Streamlit
* Pytest

## Project Outcomes

* Segmented customer populations into strategic behavioral groups
* Identified high-value customer segments with the highest revenue contribution
* Evaluated customer satisfaction through sentiment analysis
* Delivered interactive dashboard reporting for business stakeholders
* Created a reusable analytics framework suitable for future customer intelligence initiatives

## Business Insights

* Premium customer segments contribute the majority of revenue
* Customer sentiment can be used as an additional indicator of customer value
* At-risk customers can be identified for targeted retention campaigns
* Segmentation enables personalized marketing and loyalty programs
* Data-driven customer intelligence improves business decision-making

## Dashboard Preview
<img width="1633" height="1024" alt="E-Commerce Customer Segmentation Dashboard" src="https://github.com/user-attachments/assets/4e46b2fb-6e8d-4d1b-81ad-150dc73c1d0a" />

The Tableau dashboard provides an interactive view of:

* Customer segment distribution
* Revenue contribution by segment
* Sentiment performance by segment
* Customer-level drill-down analysis
* Strategic business insights for decision-makers
