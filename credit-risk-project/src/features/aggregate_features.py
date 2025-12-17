import pandas as pd

def create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregate transaction features per customer.
    """
    agg_df = df.groupby("CustomerId").agg(
        total_transaction_amount=("TransactionAmount", "sum"),
        avg_transaction_amount=("TransactionAmount", "mean"),
        transaction_count=("TransactionAmount", "count"),
        std_transaction_amount=("TransactionAmount", "std"),
    ).reset_index()

    return agg_df
