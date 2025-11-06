# feature_engineering.py

"""
Feature Engineering Module.

This script adds advanced time series features to the aggregated data:
1. Rolling Statistics: To capture local trends and volatility.
2. Seasonal Decomposition: To separate the signal into trend,
   seasonality, and residuals. Anomalies are often found in the residuals.
"""

import logging
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

def add_rolling_stats(df, window_size):
    """
    Adds rolling mean and rolling std to the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame, indexed by timestamp.
        window_size (int): The size of the rolling window.
        
    Returns:
        pd.DataFrame: DataFrame with new rolling features.
    """
    logging.info(f"Adding rolling statistics with window size {window_size}...")
    
    # We create rolling features for 'mean' and 'std'
    df[f'roll_mean_std'] = df['std'].rolling(window=window_size).mean()
    df[f'roll_std_std'] = df['std'].rolling(window=window_size).std()
    
    df[f'roll_mean_mean'] = df['mean'].rolling(window=window_size).mean()
    
    # Drop NaNs created by the rolling window
    df_rolled = df.dropna()
    
    logging.info(f"Rolling stats added. DataFrame shape: {df_rolled.shape}")
    return df_rolled

def add_seasonal_decomposition(df, period, model='additive'):
    """
    Performs seasonal decomposition and adds trend, seasonal, 
    and residual components as features.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'std' column.
        period (int): The period for the decomposition (e.g., 144 for daily).
        model (str): 'additive' or 'multiplicative'.
        
    Returns:
        pd.DataFrame: DataFrame with new decomposition features.
    """
    logging.info(f"Performing seasonal decomposition with period {period}...")
    
    # We decompose the 'std' column as it's the most indicative of failure
    try:
        decomposition = seasonal_decompose(df['std'], model=model, period=period)
        
        df['trend'] = decomposition.trend
        df['seasonal'] = decomposition.seasonal
        df['residual'] = decomposition.resid
        
        # Decomposition creates NaNs at the beginning and end
        df_decomposed = df.dropna()
        
        logging.info(f"Decomposition complete. DataFrame shape: {df_decomposed.shape}")
        return df_decomposed
        
    except ValueError as e:
        logging.error(f"Seasonal decomposition failed. Period {period} may be too large for data of shape {df.shape}. Error: {e}")
        # Return the dataframe without decomposition
        return df.dropna()