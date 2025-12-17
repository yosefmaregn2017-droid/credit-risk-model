import pandas as pd

def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract date and time features from TransactionDate.
    """
    df = df.copy()
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])

    df["transaction_hour"] = df["TransactionDate"].dt.hour
    df["transaction_day"] = df["TransactionDate"].dt.day
    df["transaction_month"] = df["TransactionDate"].dt.month
    df["transaction_year"] = df["TransactionDate"].dt.year

    return df
