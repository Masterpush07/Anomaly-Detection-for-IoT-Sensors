# 🛰️ Time Series Anomaly Detection for IoT Sensors

This project is an **end-to-end solution** for detecting anomalies in IoT sensor data using **Machine Learning and Deep Learning** techniques.  
It is developed as part of the **AI/ML Engineer (Fresher) Assignment**, demonstrating strong skills in data preprocessing, feature engineering, and anomaly detection modeling.

---

## 📘 Overview

The project simulates a **run-to-failure scenario** using the **NASA Bearing Dataset** and builds two anomaly detection models:
- 🧩 **Isolation Forest (I-Forest)** — a tree-based unsupervised anomaly detector.
- 🧠 **LSTM Autoencoder** — a deep learning model for time-series reconstruction error detection.

The pipeline is fully modular and production-ready, making it easy to extend or integrate with real-world IoT systems.

---

## 🧱 Project Structure

```

your_project_folder/
├── main.py                # The main script to run everything
├── config.py              # Configuration and parameters
├── data_loader.py         # Handles downloading, unzipping, and aggregation
├── feature_engineering.py # Adds rolling stats and seasonal decomposition
├── models.py              # Defines our I-Forest and Autoencoder models
├── .gitignore             # Tells Git to ignore data and cache files
├── README.md              # This file
├── report.docx             # The page summary report
├── requirements.txt       # Python dependencies
└── outputs/               # Auto-generated folder for all outputs
├── plots/             # Saved plots and visualizations
├── models/            # Trained models (I-Forest, LSTM Autoencoder)
└── processed_data/    # Intermediate aggregated/cleaned data

````

---

## ⚙️ How to Run

### 1️⃣ Install Dependencies
First, install the required Python libraries:
```bash
pip install -r requirements.txt
````

---

### 2️⃣ Download the Data

You have two options:

#### 🔹 Automatic Download (using Kaggle API)

If you have your Kaggle API token (`kaggle.json`) in your `~/.kaggle/` directory,
the script will automatically download the dataset:

```
vinayak123tyagi/bearing-dataset
```

#### 🔹 Manual Download

Alternatively, manually download **bearing-dataset.zip** from Kaggle and place it in the project root folder.
The script will automatically detect it and skip the download step.

---

### 3️⃣ Run the Full Pipeline

Execute the main script:

```bash
python main.py
```

The pipeline performs the following steps with detailed console logs:

1. Sets up logging at `outputs/pipeline.log`
2. Checks and processes raw data into a clean time series
3. Generates EDA plots (distributions, run-to-failure) → `outputs/plots/`
4. Performs feature engineering (rolling stats, seasonal decomposition)
5. Trains and evaluates:

   * Isolation Forest
   * LSTM Autoencoder
6. Saves trained models → `outputs/models/`
7. Generates final anomaly comparison plots → `outputs/plots/`

---

## 🧩 Key Files

| File                       | Description                                              |
| -------------------------- | -------------------------------------------------------- |
| `main.py`                  | Central orchestrator for the entire pipeline             |
| `config.py`                | Stores parameters, paths, and constants                  |
| `data_loader.py`           | Handles dataset loading, unzipping, and preprocessing    |
| `feature_engineering.py`   | Adds rolling statistics and decomposed features          |
| `models.py`                | Defines the Isolation Forest and LSTM Autoencoder models |
| `report.md` / `report.pdf` | 2–3 page final summary report with analysis              |

---

## 📊 Outputs

All generated files are automatically organized in the `outputs/` directory:

```
outputs/
├── plots/           # Visualization of trends and anomalies
├── models/          # Saved model weights and configurations
└── processed_data/  # Intermediate cleaned and aggregated datasets
```

---

## 🧠 Tech Stack

* **Python 3.x**
* **NumPy**, **Pandas**, **Matplotlib**, **Seaborn**
* **Scikit-learn**
* **TensorFlow / Keras**
* **Statsmodels**
* **Kaggle API**

---

## 🧾 Report

The detailed project summary, including methodology, models, results, and visual insights,
is available in:
* `report.docx`

---

## 📬 Author

👤 **Pushpanathan N**
📧 [GitHub Profile](https://github.com/Masterpush07)

---

## ⭐ Acknowledgements

* NASA Bearing Dataset — *for providing run-to-failure sensor data*
* Kaggle — *for dataset hosting and API integration support*

---

> “Anomalies aren’t just outliers — they’re stories waiting to be understood.” 💡

