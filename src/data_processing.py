import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# -------------------------
# Load raw CSV
# -------------------------
def load_data(path):
    """Load raw CSV data."""
    df = pd.read_csv(path)
    return df

# -------------------------
# Date feature extraction
# -------------------------
def extract_date_features(df):
    """Extract date/time features from TransactionStartTime."""
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['trans_hour'] = df['TransactionStartTime'].dt.hour
    df['trans_day'] = df['TransactionStartTime'].dt.day
    df['trans_month'] = df['TransactionStartTime'].dt.month
    df['trans_year'] = df['TransactionStartTime'].dt.year
    return df

# -------------------------
# Aggregate features per customer
# -------------------------
def aggregate_features(df):
    """Aggregate numerical features per customer."""
    agg = df.groupby('CustomerId').agg(
        total_amount=('Amount', 'sum'),
        avg_amount=('Amount', 'mean'),
        transaction_count=('Amount', 'count'),
        std_amount=('Amount', 'std')
    )
    return agg

# -------------------------
# Encode categorical variables
# -------------------------
def encode_categorical(df, categorical_cols):
    """One-hot encode categorical columns."""
    df_encoded = df.copy()
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_array = encoder.fit_transform(df_encoded[categorical_cols])
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df_encoded.index
    )
    df_encoded = df_encoded.drop(categorical_cols, axis=1)
    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
    return df_encoded

# -------------------------
# Handle missing values
# -------------------------
def handle_missing(df, strategy='mean'):
    """Fill missing numerical values."""
    df_filled = df.copy()
    num_cols = df_filled.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy=strategy)
    df_filled[num_cols] = imputer.fit_transform(df_filled[num_cols])
    return df_filled

# -------------------------
# Full preprocessing pipeline
# -------------------------
def preprocess_data_full(df, categorical_cols=None):
    """
    Full preprocessing:
    - Extract date features
    - Aggregate numeric features per customer
    - Merge latest date features
    - Handle missing values
    - Encode categorical features
    """
    # Extract date/time features
    df = extract_date_features(df)
    
    # Aggregate numeric features
    agg_df = aggregate_features(df)

    # Merge latest transaction date features
    latest_trans = df.sort_values('TransactionStartTime').groupby('CustomerId').tail(1)
    for col in ['trans_hour', 'trans_day', 'trans_month', 'trans_year']:
        agg_df[col] = latest_trans.set_index('CustomerId')[col]

    # Handle missing values
    agg_df = handle_missing(agg_df)

    # Encode categorical features if provided
    if categorical_cols:
        agg_df = encode_categorical(agg_df, categorical_cols)

    return agg_df
