# config.py
"""
Configuration file for the anomaly detection project.

This file centralizes all constants, file paths, and model parameters
for easy tuning and maintenance.
"""

from pathlib import Path

# --- Project Structure ---
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
MODEL_DIR = OUTPUT_DIR / "models"
DATA_DIR = OUTPUT_DIR / "processed_data"

# --- Data Loading ---
KAGGLE_DATASET_ID = "vinayak123tyagi/bearing-dataset"
RAW_DATA_ZIP = BASE_DIR / "bearing-dataset.zip"
RAW_DATA_PATH = BASE_DIR / "2nd_test" # Folder unzipped from the dataset
AGGREGATED_DATA_CSV = DATA_DIR / "aggregated_bearing_data.csv"
# We focus on the '2nd_test' set as it's a complete run-to-failure
DATA_SOURCE_DIR = RAW_DATA_PATH / "2nd_test"

# --- Feature Engineering ---
# We will use Bearing 1 as our primary sensor for this analysis
TARGET_COLUMN = "Bearing_1"
ROLLING_WINDOW_SIZE = 50  # 50 files = ~8.3 hours
# Seasonal decomposition period. 144 files = 24 hours (1 file per 10 min)
# We'll use a shorter period to capture daily operational cycles
SEASONAL_PERIOD = 144

# --- Model Parameters ---

# Isolation Forest
IF_N_ESTIMATORS = 100
# 'auto' is a good default, but we can guess based on our EDA plot.
# The last ~5% of the data looks anomalous.
IF_CONTAMINATION = 0.05
IF_RANDOM_STATE = 42

# LSTM Autoencoder
# We'll train on the first 70% of data, assuming it's "healthy"
AE_TRAIN_SPLIT = 0.7
AE_TIME_STEPS = 10      # Look at sequences of 10 data points
AE_EPOCHS = 50
AE_BATCH_SIZE = 32
AE_LSTM_UNITS = 64
AE_VALIDATION_SPLIT = 0.1
# Anomalies are points with a reconstruction error in the top 1%
# of the *training* data's error. This is a statistical threshold.
AE_ANOMALY_THRESHOLD_PCT = 0.99