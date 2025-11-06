# README.md

Project: Time Series Anomaly Detection for IoT Sensors

This project is an end-to-end solution for detecting anomalies in IoT sensor data, as per the AI/ML Engineer (Fresher) Assignment. It uses the NASA Bearing Dataset to simulate a "run-to-failure" scenario and builds two models (Isolation Forest and an LSTM Autoencoder) to detect anomalies.

The code is structured as a modular Python project for clarity, maintainability, and production-readiness.

Project Structure

your_project_folder/
├── main.py               # The main script to run everything
├── config.py             # Configuration and parameters
├── data_loader.py        # Handles downloading, unzipping, and aggregation
├── feature_engineering.py # Adds rolling stats and seasonal decomposition
├── models.py             # Defines our I-Forest and Autoencoder models
├── README.md             # This file
├── report.md             # The final 2-3 page summary report
├── requirements.txt      # Project dependencies
├── outputs/              # Auto-generated folder for all outputs
│   ├── plots/
│   ├── models/
│   └── processed_data/


How to Run

Install Dependencies:

pip install -r requirements.txt


Download the Data:

This script assumes you have a kaggle.json API token in your ~/.kaggle/ directory.

The main.py script will automatically try to download the dataset: vinayak123tyagi/bearing-dataset

If you have already downloaded the bearing-dataset.zip file, simply place it in the root of this project folder.

Run the Pipeline:
Execute the main script from your terminal.

python main.py


The script will:

Set up logging.

Check for and process the raw data into a clean, 984-row time series.

Generate and save EDA plots (distributions, run-to-failure) to outputs/plots/.

Perform advanced feature engineering (rolling stats, seasonal decomposition) and save plots.

Train, evaluate, and save both the Isolation Forest and LSTM Autoencoder models to outputs/models/.

Generate and save the final anomaly comparison plots to outputs/plots/.

Key Files

main.py: The central orchestrator. Run this file.

report.md: The 2-3 page summary document explaining the approach, models, and findings.