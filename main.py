"""
End-to-End Time Series Anomaly Detection Pipeline

This is the main orchestrator script. Run this file to execute
the full pipeline:
1.  Setup logging and directories
2.  Load and aggregate raw data (using data_loader.py)
3.  Perform Exploratory Data Analysis (EDA) and save plots
4.  Engineer features (using feature_engineering.py)
5.  Train and evaluate both models (using models.py)
6.  Save final models, plots, and results
"""

import logging
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Import our custom modules
import config
import data_loader
import feature_engineering
import models

# --- 1. Setup Logging and Directories ---

# Create output directories if they don't exist
config.PLOT_DIR.mkdir(parents=True, exist_ok=True)
config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
config.DATA_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.OUTPUT_DIR / "pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.info("--- Starting Anomaly Detection Pipeline ---")

# --- Helper Function for Plotting ---
def save_plot(fig, filename, plot_dir=config.PLOT_DIR):
    """Helper to save matplotlib figures."""
    try:
        path = plot_dir / filename
        fig.savefig(path, bbox_inches='tight', dpi=150)
        logging.info(f"Plot saved: {path}")
    except Exception as e:
        logging.error(f"Failed to save plot {filename}. Error: {e}")
    plt.close(fig)

# --- 2. Run the Main Pipeline ---

def run_pipeline():
    """Executes the full data-to-model pipeline."""
    
    # --- Step 1: Load and Aggregate Data ---
    logging.info("=== Step 1: Data Loading and Aggregation ===")
    try:
        # Download/unzip if necessary
        # api = data_loader.setup_kaggle_api() # <-- REMOVED THIS LINE
        
        # UPDATED THIS LINE: Removed the 'api' argument
        if not data_loader.download_and_unzip(config.KAGGLE_DATASET_ID, config.RAW_DATA_ZIP, config.RAW_DATA_PATH):
            if not config.RAW_DATA_PATH.exists():
                logging.error("Data loading failed and no existing data found. Exiting.")
                return
            else:
                logging.warning("Data download failed. Attempting to use existing local data.")

        # Aggregate the 20M+ rows into a 984-row time series
        features_df = data_loader.aggregate_data(
            config.DATA_SOURCE_DIR, 
            config.AGGREGATED_DATA_CSV,
            config.TARGET_COLUMN
        )
        if features_df is None:
            logging.error("Data aggregation failed. Exiting.")
            return
    except Exception as e:
        logging.error(f"An error occurred during data loading: {e}", exc_info=True)
        return
        
    logging.info(f"Aggregated data loaded. Shape: {features_df.shape}")
    logging.info(features_df.head())

    # --- Step 2: Exploratory Data Analysis (EDA) ---
    logging.info("=== Step 2: Exploratory Data Analysis ===")
    
    # Plot 1: Run-to-Failure (The "Money Plot")
    fig, ax = plt.subplots(figsize=(15, 6))
    features_df['std'].plot(ax=ax, title='Run-to-Failure: Bearing Vibration (Std. Dev) Over Time')
    ax.set_ylabel('Standard Deviation')
    ax.set_xlabel('Timestamp')
    save_plot(fig, "1_eda_run_to_failure.png")

    # Plot 2: Feature Distributions
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Distribution of Aggregated Features (Healthy vs. Failure)', fontsize=16)
    plot_features = ['mean', 'std', 'min', 'max', 'skew', 'kurtosis']
    
    # Split for visualization
    split_point = int(len(features_df) * config.AE_TRAIN_SPLIT)
    df_healthy = features_df.iloc[:split_point]
    df_fail = features_df.iloc[split_point:]

    for i, col in enumerate(plot_features):
        ax = axes.flatten()[i]
        sns.kdeplot(df_healthy[col], ax=ax, label='Healthy', shade=True)
        sns.kdeplot(df_fail[col], ax=ax, label='Failing', shade=True)
        ax.set_title(f'Distribution of {col}')
        ax.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_plot(fig, "2_eda_feature_distributions.png")

    # --- Step 3: Feature Engineering ---
    logging.info("=== Step 3: Feature Engineering ===")
    
    # Add Rolling Stats
    df_rolled = feature_engineering.add_rolling_stats(
        features_df, 
        config.ROLLING_WINDOW_SIZE
    )
    
    # Add Seasonal Decomposition
    df_final_features = feature_engineering.add_seasonal_decomposition(
        df_rolled,
        config.SEASONAL_PERIOD
    )
    
    # Plot 3: Seasonal Decomposition
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    fig.suptitle("Seasonal Decomposition of Vibration (Std. Dev)", fontsize=16)
    df_final_features['std'].plot(ax=ax1, title='Observed')
    df_final_features['trend'].plot(ax=ax2, title='Trend')
    df_final_features['seasonal'].plot(ax=ax3, title='Seasonal')
    df_final_features['residual'].plot(ax=ax4, title='Residual', marker='.', linestyle='none')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_plot(fig, "3_feat_seasonal_decomposition.png")
    
    # This is our final set of features for the models
    # We drop timestamp and the original 'std' (since it's decomposed)
    model_input_df = df_final_features.drop(columns=['std', 'trend', 'seasonal'])
    # Also drop 'mean' and other raw stats, focusing on residuals and rolling stats
    model_input_df = model_input_df[['residual', 'roll_mean_std', 'roll_std_std', 'roll_mean_mean']]
    logging.info(f"Final feature set for models: {model_input_df.columns.tolist()}")

    # --- Step 4: Model Training ---
    logging.info("=== Step 4: Model Training ===")

    # Approach 1: Isolation Forest
    if_model_path = config.MODEL_DIR / "isolation_forest.joblib"
    if_preds, if_scores = models.train_isolation_forest(model_input_df, if_model_path)
    df_final_features['if_anomaly'] = if_preds  # -1 for anomaly
    df_final_features['if_score'] = if_scores

    # Approach 2: LSTM Autoencoder
    ae_model_path = config.MODEL_DIR / "lstm_autoencoder.keras"
    autoencoder, scaler, X_train = models.train_autoencoder(model_input_df, ae_model_path)
    
    # --- Step 5: Model Evaluation & Visualization ---
    logging.info("=== Step 5: Model Evaluation ===")

    # Get Autoencoder anomalies
    ae_mae, ae_threshold, ae_preds = models.get_autoencoder_anomalies(
        autoencoder, scaler, model_input_df
    )
    df_final_features['ae_mae'] = ae_mae
    df_final_features['ae_anomaly'] = ae_preds # 1 for anomaly

    # Plot 4: Autoencoder Reconstruction Error
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(df_final_features.index, df_final_features['ae_mae'], label='Reconstruction Error (MAE)')
    ax.axhline(ae_threshold, color='r', linestyle='--', label=f'Anomaly Threshold ({ae_threshold:.4f})')
    ax.set_title('Autoencoder Anomaly Detection (Error Plot)')
    ax.set_xlabel('Timestamp')
    ax.legend()
    save_plot(fig, "4_eval_autoencoder_error.png")

    # Plot 5: Final Model Comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    # Get anomaly points for plotting
    if_anomalies = df_final_features[df_final_features['if_anomaly'] == -1]
    ae_anomalies = df_final_features[df_final_features['ae_anomaly'] == 1]
    
    # Plot 1: Isolation Forest Results
    ax1.plot(df_final_features.index, df_final_features['std'], label='Original Signal (std)')
    ax1.scatter(if_anomalies.index, if_anomalies['std'], color='red', label='IF Anomaly', s=50)
    ax1.set_title('Approach 1: Isolation Forest Detections')
    ax1.legend()
    
    # Plot 2: Autoencoder Results
    ax2.plot(df_final_features.index, df_final_features['std'], label='Original Signal (std)')
    ax2.scatter(ae_anomalies.index, ae_anomalies['std'], color='orange', label='AE Anomaly', s=50)
    ax2.set_title('Approach 2: LSTM Autoencoder Detections')
    ax2.set_xlabel('Timestamp')
    ax2.legend()
    
    save_plot(fig, "5_eval_model_comparison.png")

    logging.info("--- Pipeline Finished Successfully ---")
    logging.info(f"All outputs saved to: {config.OUTPUT_DIR}")

if __name__ == "__main__":
    run_pipeline()