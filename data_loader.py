"""
Data Loading and Aggregation Module.

This script handles:
1. Downloading and unzipping the raw Kaggle dataset.
   (Lazily imports Kaggle API to avoid auth errors if data is local)
2. Processing the 20M+ rows from 984 files into a clean, 
   984-row aggregated time series DataFrame.
"""

import os
import zipfile
import logging
from glob import glob
import pandas as pd

# We DO NOT import Kaggle API here to prevent auto-authentication
import config

def setup_kaggle_api():
    """Sets up the Kaggle API, handling authentication."""
    try:
        # --- THIS IS THE MOVED IMPORT ---
        from kaggle.api.kaggle_api_extended import KaggleApi
        # ---------------------------------
        api = KaggleApi()
        api.authenticate()
        return api
    except OSError as e:
        logging.error(f"Kaggle API authentication failed. {e}")
        logging.error("Please ensure 'kaggle.json' is in C:\\Users\\pushp\\.kaggle\\")
        logging.error("Or, to bypass, manually download 'bearing-dataset.zip' to the project folder.")
        return None
    except ImportError:
        logging.error("Kaggle package not found. Please run 'pip install kaggle'")
        return None


def download_and_unzip(dataset_id, zip_path, extract_path):
    """Downloads and unzips the dataset if not already present."""
    if zip_path.exists():
        logging.info(f"Dataset zip file found locally: {zip_path}")
    else:
        # --- Now we call the setup function, only when needed ---
        api = setup_kaggle_api()
        if api is None:
            return False # API setup failed, can't download
            
        try:
            logging.info(f"Downloading dataset '{dataset_id}' to {zip_path}...")
            # Note: The original code had a bug, downloading to parent. Fixed.
            api.dataset_download_files(dataset_id, path=zip_path.parent, quiet=False, force=False, unzip=False)
            
            # The API call above might download it as 'bearing-dataset.zip'
            # Let's rename it if it downloaded with a different name (e.g. archive.zip)
            # This is a common Kaggle API issue
            default_zip_name = zip_path.parent / (dataset_id.split('/')[1] + '.zip')
            if not zip_path.exists() and default_zip_name.exists():
                default_zip_name.rename(zip_path)

            logging.info("Download complete.")
        except Exception as e:
            logging.error(f"Failed to download dataset. Error: {e}")
            return False

    if extract_path.exists():
        logging.info(f"Data already extracted to: {extract_path}")
    else:
        try:
            logging.info(f"Unzipping {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(path=extract_path.parent) # Extracts into the root
            logging.info(f"Unzipped successfully to {extract_path.parent}")
        except Exception as e:
            logging.error(f"Failed to unzip file '{zip_path}'. Error: {e}")
            return False
    return True

def aggregate_data(raw_data_dir, output_csv, target_column):
    """
    Aggregates the raw 10-minute snapshot files into a single
    time series DataFrame with statistical features.
    
    Each 10-min file (20,480 readings) becomes ONE row.
    """
    if output_csv.exists():
        logging.info(f"Aggregated data already exists: {output_csv}")
        try:
            df = pd.read_csv(output_csv, parse_dates=['timestamp'])
            # Check if 'timestamp' is the index, if not, set it.
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            return df
        except Exception as e:
            logging.warning(f"Could not load existing CSV. Re-processing. Error: {e}")

    logging.info(f"Starting data aggregation from: {raw_data_dir}")
    all_file_paths = glob(str(raw_data_dir / '*'))
    if not all_file_paths:
        logging.error(f"No files found in {raw_data_dir}. Check 'DATA_SOURCE_DIR' in config.py")
        return None
    
    logging.info(f"Found {len(all_file_paths)} files to process.")
    
    all_data_list = []
    for path in all_file_paths:
        try:
            # Read the raw file
            df_file = pd.read_table(path, sep='\t', header=None)
            
            # Use only the target bearing column (e.g., column 0 for Bearing 1)
            col_index = int(target_column.split('_')[-1]) - 1
            sensor_data = df_file.iloc[:, col_index]
            
            # Get timestamp from filename
            timestamp_str = os.path.basename(path)
            timestamp = pd.to_datetime(timestamp_str, format='%Y.%m.%d.%H.%M.%S')
            
            # Calculate statistical features for this 10-min file
            features = {
                'timestamp': timestamp,
                'mean': sensor_data.mean(),
                'std': sensor_data.std(),
                'min': sensor_data.min(),
                'max': sensor_data.max(),
                'skew': sensor_data.skew(),
                'kurtosis': sensor_data.kurtosis()
            }
            all_data_list.append(features)
        except Exception as e:
            logging.warning(f"Failed to process file {path}. Error: {e}")

    if not all_data_list:
        logging.error("No data was successfully processed.")
        return None

    # Create the final aggregated DataFrame
    features_df = pd.DataFrame(all_data_list)
    features_df = features_df.sort_values(by='timestamp').set_index('timestamp')
    
    # Save the processed data
    logging.info(f"Aggregation complete. Saving to {output_csv}")
    features_df.to_csv(output_csv)
    
    return features_df