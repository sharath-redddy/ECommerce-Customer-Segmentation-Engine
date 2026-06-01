import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger("Pipeline")

class DataPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.raw_df = pd.DataFrame()

    def load_data(self) -> pd.DataFrame:
        url = self.config['data']['stream_url']
        logger.info(f"Downloading data from: {url}")
        self.raw_df = pd.read_csv(url, on_bad_lines='skip')
        os.makedirs(os.path.dirname(self.config['data']['raw_path']), exist_ok=True)
        self.raw_df.to_csv(self.config['data']['raw_path'], index=False)
        return self.raw_df

    def _clean_price(self, val) -> float:
        try:
            return float(str(val).split('-')[0].replace('£','').replace('$','').strip())
        except:
            return np.nan

    def _extract_rating(self, val) -> float:
        try:
            if 'out of 5 stars' in str(val):
                return float(str(val).split('out of 5 stars')[0].split('//')[-1].strip())
            return np.nan
        except:
            return np.nan

    def _parse_dates(self, val):
        try:
            date_str = str(val).split('//')[0].replace('on', '').strip()
            date_str = date_str.replace('Nov.', 'November').replace('Dec.', 'December')
            for fmt in ('%d %B %Y', '%d %b %Y', '%B %d, %Y'):
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            return pd.NaT
        except:
            return pd.NaT

    def clean_and_filter(self) -> pd.DataFrame:
        if self.raw_df.empty:
            raise ValueError("Raw dataset is empty.")
        df = self.raw_df.copy()
        df['Price'] = df['price'].apply(self._clean_price)
        df['Star_Rating'] = df['customer_reviews'].apply(self._extract_rating)
        df['Review_Date'] = df['customer_reviews'].apply(self._parse_dates)
        df['Review_Text'] = df['customer_reviews'].astype(str).str.strip()

        df['Price'] = df['Price'].fillna(df['Price'].median() if not df['Price'].isnull().all() else 49.99)
        df['Star_Rating'] = df['Star_Rating'].fillna(4.0)

        valid_dates = df['Review_Date'].dropna()
        if not valid_dates.empty:
            max_date = valid_dates.max()
            start_date = max_date - pd.DateOffset(months=self.config['pipeline']['rolling_months'])
            df = df[(df['Review_Date'] >= start_date) & (df['Review_Date'] <= max_date)].copy()
            logger.info(f"Timeline synchronized: {start_date.date()} to {max_date.date()}")

        df = df.rename(columns={'uniq_id': 'User_ID'})
        return df

    def aggregate_customers(self, df: pd.DataFrame) -> pd.DataFrame:
        max_date = df['Review_Date'].max() if 'Review_Date' in df.columns and not df['Review_Date'].isnull().all() else datetime.now()
        df['Recency_Days'] = df['Review_Date'].apply(lambda x: (max_date - x).days if pd.notnull(x) else 0)

        customer_df = df.groupby("User_ID").agg({\n            "Recency_Days": "min",
            "price": "count",
            "Price": "sum",
            "Star_Rating": "mean"
        }).rename(columns={
            "price": "Frequency",
            "Price": "Monetary",
            "Star_Rating": "Average_Rating"
        })

        text_mapping = df.groupby("User_ID")["Review_Text"].apply(lambda x: " ".join(x)).to_dict()
        customer_df["Aggregated_Reviews"] = customer_df.index.map(text_mapping)
        return customer_df.reset_index()
