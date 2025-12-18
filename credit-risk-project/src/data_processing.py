# src/data_processing.py (FINAL, CORRECTED VERSION for Pipeline/WoE Integration)

import pandas as pd
import numpy as np
import os
import logging
from typing import List, Tuple, Dict, Any
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Added OneHotEncoder for non-WoE use
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Utility Functions ---
# (Load Data, Extract Time Features, Calculate Recency remain the same)

def load_data(file_path: str) -> pd.DataFrame | None:
    """Loads transaction data from a CSV file."""
    if not os.path.exists(file_path):
        logging.error(f"File not found at: {file_path}. Please check your 'data/raw/' directory.")
        return None
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data successfully loaded from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"An unexpected error occurred during data loading: {e}")
        return None

def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Converts TransactionStartTime to datetime, extracts temporal features, and handles redundancy."""
    if 'TransactionStartTime' in df.columns:
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.dayofweek
        logging.info("Temporal features (Hour and Day) extracted.")
    
    if 'Amount' in df.columns:
        df.drop(columns=['Amount'], inplace=True)
        logging.info("Redundant 'Amount' column dropped.")
    
    return df

def calculate_recency(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the Recency metric (days since last transaction) for each customer."""
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    recency_df = df.groupby('CustomerId').agg(
        LastTransaction=('TransactionStartTime', 'max')
    ).reset_index()
    recency_df['Recency'] = (snapshot_date - recency_df['LastTransaction']).dt.days
    recency_df.drop(columns=['LastTransaction'], inplace=True)
    logging.info(f"Recency calculated for {len(recency_df)} customers.")
    return recency_df

def aggregate_features(df: pd.DataFrame, recency_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transactional data to the CustomerId level.
    (MODIFIED to include most frequent CATEGORIES for later WoE processing)
    """
    agg_funcs = {
        'Value': ['sum', 'mean', 'std', 'max'],
        'TransactionHour': ['mean', 'std', 'max'],
        'TransactionDay': ['mean', 'max'],
        'TransactionId': 'count'
    }

    # Identify the most frequent category for key categorical features (used for WoE)
    categorical_cols = ['ProviderId', 'ProductCategory', 'ChannelId', 'PricingStrategy', 'CurrencyCode', 'CountryCode']
    
    # Calculate the mode (most frequent) for all key categoricals
    mode_features = df.groupby('CustomerId')[categorical_cols].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else 'MISSING'
    ).reset_index()
    
    # Calculate numerical aggregations
    customer_df = df.groupby('CustomerId').agg(agg_funcs)

    # Flatten column names and rename Frequency
    customer_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                           for col in customer_df.columns.values]
    customer_df.rename(columns={'TransactionId_count': 'Frequency'}, inplace=True)
    customer_df = customer_df.reset_index()
    
    # Merge all components
    final_df = customer_df.merge(recency_df, on='CustomerId', how='left')
    final_df = final_df.merge(mode_features, on='CustomerId', how='left') # MERGE RAW CATEGORIES
    
    logging.info(f"Features aggregated and merged with Recency and raw categorical features. Final Shape: {final_df.shape}")
    return final_df

# --- 2. Custom Scikit-learn Transformer (WoE) ---

class WoETransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to calculate and apply Weight of Evidence (WoE) transformation.
    """
    def __init__(self, target_column: str = 'is_high_risk', smooth: float = 0.5):
        self.woe_map = {}
        self.target_column = target_column
        self.smooth = smooth

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # We must receive X and y separately, as expected by ColumnTransformer/Pipeline standard.
        
        # NOTE: We assume X only contains the categorical columns that need WOE.
        
        for col in X.columns:
            # Handle missing categories by treating them as a new category
            df_temp = X[[col]].copy()
            df_temp['target'] = y
            
            # Use 'MISSING' as the constant fill value for imputation later
            df_temp[col].fillna('MISSING', inplace=True)
            
            # Calculate counts and rates
            group = df_temp.groupby(col)['target']
            total_bads = group.sum().sum()
            total_goods = group.count().sum() - total_bads

            bad_rate = group.sum()
            good_rate = group.count() - bad_rate
            
            # Apply smoothing
            P_N1 = (bad_rate + self.smooth) / (total_bads + self.smooth)
            P_N0 = (good_rate + self.smooth) / (total_goods + self.smooth)

            # Calculate WoE
            woe = np.log(P_N0 / P_N1)
            self.woe_map[col] = woe.to_dict()
            
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_transformed = X.copy()
        
        for col in X.columns:
            if col in self.woe_map:
                # Fill NaNs with 'MISSING' before mapping (to handle missing values in input)
                X_transformed[col].fillna('MISSING', inplace=True)
                
                # Map the WoE values. Use the median WoE from the training set for unseen categories.
                median_woe = np.median(list(self.woe_map[col].values()))
                
                # Apply mapping and fill any remaining NaNs (for unseen categories) with the median WoE
                X_transformed[col] = X_transformed[col].map(self.woe_map[col]).fillna(median_woe)
                X_transformed.rename(columns={col: f'{col}_WOE'}, inplace=True)
            
        return X_transformed

# --- 3. Full Preprocessing Pipeline Definition ---

def get_preprocessing_pipeline(df: pd.DataFrame, target_col: str):
    """
    Defines the full ColumnTransformer preprocessing pipeline which handles:
    1. Numerical Scaling (Median Imputer + StandardScaler)
    2. Categorical WoE Encoding (using WoETransformer)
    
    Returns the fitted preprocessor object and the feature lists.
    """
    # 1. Identify feature types from the aggregated DataFrame
    
    # Features that need Numerical Treatment
    numerical_cols = [
        col for col in df.columns 
        if df[col].dtype in ['int64', 'float64'] and col not in ['CustomerId', target_col]
    ]

    # Features that need Categorical Treatment (WoE)
    categorical_cols = [
        col for col in df.columns 
        if df[col].dtype in ['object', 'category'] and col not in ['CustomerId', target_col]
    ]

    # 2. Define Numerical Pipeline
    numerical_pipeline = make_pipeline(
        SimpleImputer(strategy='median'), # EDA Insight: Robust to outliers/skewness
        StandardScaler()                  # EDA Insight: Required for distance-based models (K-Means)
    )
    
    # 3. Define Categorical Pipeline
    # NOTE: We use SimpleImputer(constant='MISSING') to explicitly handle NaNs in categorical columns
    # and pass it directly to the WoETransformer.
    categorical_pipeline = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='MISSING'),
        WoETransformer(target_column=target_col)
    )

    # 4. Define Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols) # WoE integration point
        ],
        remainder='passthrough' # Keep CustomerId (index)
    )
    
    logging.info(f"Full ColumnTransformer defined: {len(numerical_cols)} numerical features and {len(categorical_cols)} categorical features.")
    
    return preprocessor, numerical_cols, categorical_cols

# --- 4. Orchestration Function (Simplified, used for initial data prep) ---

def process_data_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the pre-aggregation feature engineering steps (Task 3).
    NOTE: This is run BEFORE Task 4 (Target Creation) and Task 5 (Final Preprocessing/Training).
    """
    df = extract_time_features(df)
    recency_df = calculate_recency(df)
    customer_df = aggregate_features(df, recency_df)
    
    # NO SCALING/WOE is applied here yet, as the target is not created until Task 4.
    logging.info("Task 3 (Feature Aggregation) complete. Ready for Target Creation (Task 4).")
    return customer_df