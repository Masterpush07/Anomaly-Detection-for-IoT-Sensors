"""
Anomaly Detection Models Module.

This script defines, trains, and evaluates our two models:
1. Approach 1: Isolation Forest (Unsupervised)
2. Approach 2: LSTM Autoencoder (Deep Learning)
"""

import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping

import config

# --- Helper Functions for Autoencoder ---

def _create_sequences(data, time_steps):
    """Helper to convert 2D data into 3D sequences for LSTM."""
    X = []
    for i in range(len(data) - time_steps + 1):
        X.append(data[i:(i + time_steps)])
    return np.array(X)

def _build_autoencoder(timesteps, n_features, lstm_units):
    """Defines the Keras LSTM Autoencoder architecture."""
    inputs = Input(shape=(timesteps, n_features))
    # Encoder
    encoder = LSTM(lstm_units, activation='relu', return_sequences=False)(inputs)
    repeater = RepeatVector(timesteps)(encoder)
    # Decoder
    decoder = LSTM(lstm_units, activation='relu', return_sequences=True)(repeater)
    output = Dense(n_features, activation='linear')(decoder)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mae')
    logging.info("LSTM Autoencoder model built.")
    model.summary(print_fn=logging.info)
    return model

# --- Approach 1: Isolation Forest ---

def train_isolation_forest(df_features, save_path):
    """
    Trains an Isolation Forest model and returns anomaly scores.
    
    Args:
        df_features (pd.DataFrame): DataFrame with all features.
        save_path (Path): Path to save the trained model.
        
    Returns:
        np.array: Anomaly predictions (-1 for anomaly, 1 for normal).
    """
    logging.info("Starting Isolation Forest training...")
    model_if = IsolationForest(
        n_estimators=config.IF_N_ESTIMATORS,
        contamination=config.IF_CONTAMINATION,
        random_state=config.IF_RANDOM_STATE,
        n_jobs=-1
    )
    
    model_if.fit(df_features)
    logging.info("Isolation Forest training complete.")
    
    # Save the model
    try:
        joblib.dump(model_if, save_path)
        logging.info(f"Isolation Forest model saved to {save_path}")
    except Exception as e:
        logging.error(f"Failed to save Isolation Forest model. Error: {e}")
        
    # Get predictions
    predictions = model_if.predict(df_features)
    # Get anomaly scores (lower = more anomalous)
    scores = model_if.decision_function(df_features)
    
    return predictions, scores

# --- Approach 2: LSTM Autoencoder ---

def train_autoencoder(df_features, save_path):
    """
    Trains the LSTM Autoencoder *only* on healthy data.
    
    Args:
        df_features (pd.DataFrame): DataFrame with all features.
        save_path (Path): Path to save the trained model.
        
    Returns:
        tuple: (trained_model, scaler, training_data_sequences)
    """
    logging.info("Starting LSTM Autoencoder training...")
    
    # 1. Split data into "healthy" (train) and "all" (test)
    split_index = int(len(df_features) * config.AE_TRAIN_SPLIT)
    df_train = df_features.iloc[:split_index]
    
    # 2. Scale the data
    # IMPORTANT: Fit the scaler ONLY on the training data
    scaler = StandardScaler()
    data_train_scaled = scaler.fit_transform(df_train)
    
    # 3. Create sequences
    X_train = _create_sequences(data_train_scaled, config.AE_TIME_STEPS)
    
    if X_train.shape[0] == 0:
        logging.error("Training data is too small to create sequences. Adjust TIME_STEPS or data split.")
        return None, None, None
        
    logging.info(f"Created training sequences. Shape: {X_train.shape}")
    
    # 4. Build and train the model
    n_features = X_train.shape[2]
    autoencoder = _build_autoencoder(config.AE_TIME_STEPS, n_features, config.AE_LSTM_UNITS)
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
    
    history = autoencoder.fit(
        X_train, X_train,
        epochs=config.AE_EPOCHS,
        batch_size=config.AE_BATCH_SIZE,
        validation_split=config.AE_VALIDATION_SPLIT,
        callbacks=[early_stopping],
        shuffle=False
    )
    
    logging.info("Autoencoder training complete.")
    
    # Save the model and scaler
    try:
        autoencoder.save(save_path)
        logging.info(f"Autoencoder model saved to {save_path}")
        scaler_path = save_path.parent / "autoencoder_scaler.joblib"
        joblib.dump(scaler, scaler_path)
        logging.info(f"Autoencoder scaler saved to {scaler_path}")
    except Exception as e:
        logging.error(f"Failed to save Autoencoder model/scaler. Error: {e}")
        
    return autoencoder, scaler, X_train

def get_autoencoder_anomalies(autoencoder, scaler, full_data):
    """
    Uses a trained autoencoder to get reconstruction error and 
    detect anomalies in the *full* dataset.
    
    Returns:
        tuple: (mae, threshold, predictions)
    """
    if autoencoder is None:
        return None, None, None

    logging.info("Calculating reconstruction error for full dataset...")
    
    # 1. Scale *all* data using the *fitted* scaler
    data_scaled = scaler.transform(full_data)
    
    # 2. Create sequences for all data
    X_all_sequences = _create_sequences(data_scaled, config.AE_TIME_STEPS)
    
    # 3. Get model's reconstructions
    reconstructions = autoencoder.predict(X_all_sequences)
    
    # 4. Calculate Mean Absolute Error (MAE) for each sequence
    mae = np.mean(np.abs(X_all_sequences - reconstructions), axis=(1, 2))
    
    # 5. Calculate reconstruction error on the training data to find a threshold
    split_index = int(len(full_data) * config.AE_TRAIN_SPLIT)
    train_mae = mae[:(split_index - config.AE_TIME_STEPS + 1)]
    
    # 6. Set threshold statistically
    threshold = np.quantile(train_mae, config.AE_ANOMALY_THRESHOLD_PCT)
    logging.info(f"Calculated anomaly threshold (99th percentile): {threshold}")
    
    # 7. Get final predictions
    predictions = (mae > threshold).astype(int) # 1 = anomaly, 0 = normal
    
    # Pad the beginning of the arrays (which couldn't be sequenced)
    padding = [None] * (config.AE_TIME_STEPS - 1)
    mae_padded = np.concatenate([padding, mae])
    predictions_padded = np.concatenate([padding, predictions])
    
    return mae_padded, threshold, predictions_padded