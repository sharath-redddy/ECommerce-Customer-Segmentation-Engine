import pandas as pd
from textblob import TextBlob

class SentimentAnalyzer:
    def calculate_sentiment(self, customer_df: pd.DataFrame) -> pd.DataFrame:
        customer_df['Average_Sentiment'] = customer_df['Aggregated_Reviews'].apply(
            lambda text: TextBlob(str(text)).sentiment.polarity if str(text).strip() else 0.0
        )
        customer_df['Sentiment_Label'] = customer_df['Average_Sentiment'].apply(
            lambda s: "Positive" if s > 0.05 else ("Negative" if s < -0.05 else \"Neutral\")
        )
        return customer_df
