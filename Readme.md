# Interactive Time Series Forecasting App

This project is a Streamlit application for end-to-end time series forecasting and analysis. It allows users to:

- **Upload** a CSV dataset containing a time series.
- **Visualize** the original time series with rolling statistics (mean and standard deviation).
- **Decompose** the series into Observed, Trend, Seasonal, and Residual components (Additive/Multiplicative).
- **Forecast** future values using ARIMA, ETS (Exponential Smoothing), or Prophet models.
- **Evaluate & Compare** models using performance metrics (MSE, RMSE, MAE, MAPE).
- **Export** trained models in `.sav` format for later use.

---

## Table of Contents

- [Interactive Time Series Forecasting App](#interactive-time-series-forecasting-app)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the App Locally](#running-the-app-locally)
  - [Usage](#usage)
    - [1. Upload Dataset](#1-upload-dataset)
    - [2. Visualization](#2-visualization)
    - [3. Seasonal Decomposition](#3-seasonal-decomposition)
    - [4. Forecasting](#4-forecasting)
    - [5. Performance Comparison](#5-performance-comparison)
  - [Project Structure](#project-structure)
  - [Deployment](#deployment)
  - [Contributing](#contributing)
  - [License](#license)
- [Credit Card Fraud Detection - End-to-End MLOps Project](#credit-card-fraud-detection---end-to-end-mlops-project)
  - [Problem Statement](#problem-statement)
  - [Project Architecture](#project-architecture)
  - [Key Features](#key-features)
  - [EDA \& Insights](#eda--insights)
  - [Machine Learning Pipeline](#machine-learning-pipeline)
  - [Model Evaluation (Best Models)](#model-evaluation-best-models)
  - [MLOps Stack](#mlops-stack)
  - [Installation](#installation-1)
  - [Future Improvements](#future-improvements)
  - [Acknowledgments](#acknowledgments)
  - [Contact](#contact)

---

## Features

- **Interactive Widgets:** Pythonic code using Streamlit widgets (`file_uploader`, `selectbox`, `number_input`, `radio`, `button`).
- **Plotly Integration:** Interactive, zoomable graphs for all visualizations and forecasts.
- **Session State:** Persistent performance metrics across multiple model runs.
- **Model Export:** Trained models serialized via `joblib` and downloadable in `.sav` format.
- **Conditional Formatting:** Performance table highlights best (green), worst (red), and intermediate (yellow) models by RMSE.

---

## Prerequisites

- Python 3.7 or newer
- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Plotly](https://plotly.com/python/)
- [statsmodels](https://www.statsmodels.org/)
- [prophet](https://facebook.github.io/prophet/) (optional, for Prophet model)
- [joblib](https://joblib.readthedocs.io/)

Ensure you have a compatible NVIDIA CUDA toolkit installed if using GPU-accelerated libraries.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name/src
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the App Locally

From the `src` directory, run:

```bash
streamlit run app.py
```

This will launch a local server (usually at `http://localhost:8501`). Open the URL in your browser.

---

## Usage

### 1. Upload Dataset

- Click **Browse files** to upload a CSV with at least two columns: one datetime column and one numeric target column.

### 2. Visualization

- **Rolling Statistics:** Choose a rolling window size to display the time series with its rolling mean and standard deviation.

### 3. Seasonal Decomposition

- Select **Additive** or **Multiplicative** decomposition.
- Enter the seasonal period (e.g., 7 for weekly seasonality).
- View the four-panel decomposition plot.

### 4. Forecasting

- Choose a model from **ARIMA**, **ETS**, or **Prophet**.
- Enter the forecast horizon (number of future periods).
- Click **Train and Forecast** to see:
  - Interactive forecast plot.
  - Performance metrics table update.
  - Download link for the trained model (`.sav`).

### 5. Performance Comparison

- The bottom table lists all trained models with metrics:
  - MSE, RMSE, MAE, MAPE.
  - Best RMSE row is dark green (#337142).
  - Worst RMSE row is dark red (#811414).
  - Others are amber (#8b7400).
- Click **Export Model** to download the serialized model.

---

## Project Structure

```
├── README.md
├── requirements.txt
└── src
    └── app.py
```

- **app.py:** Main Streamlit application script.
- **requirements.txt:** List of Python packages required.
- **README.md:** This documentation file.

---

## Deployment

To deploy on Streamlit Cloud:

1. Push your code to GitHub.
2. In Streamlit Cloud, click **New app** and connect your GitHub repo.
3. Set the `main.py` (or `src/app.py`) entrypoint and branch.
4. Provide `requirements.txt` in the repo root.
5. Deploy — Streamlit will install dependencies and launch your app.

---

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to:

- Fork the repository.
- Create a new branch for your feature/bug fix.
- Submit a pull request with a clear description.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

# Credit Card Fraud Detection - End-to-End MLOps Project

This project focuses on detecting fraudulent credit card transactions using machine learning techniques, versioned data pipelines, and MLOps best practices.

---

## Problem Statement

Fraudulent transactions represent a major financial risk for banks, customers, and businesses.  This project builds a robust end-to-end system to:
- Analyze transaction patterns
- Handle highly **imbalanced data**
- Build and optimize various **classification models**
- Track experiments, version datasets, and automate pipelines

---

## Project Architecture

```
📦 project/
 ┣ 📂 data/
 ┃ ┣ 📂 raw/               # Original dataset (Kaggle)
 ┃ ┣ 📂 processed/         # Cleaned and feature-engineered datasets
 ┣ 📂 notebooks/          # EDA, visualization, modeling experiments
 ┣ 📂 src/                # Python scripts: cleaning, feature engineering, modeling
 ┣ 📂 models/             # Saved models
 ┣ 📄 requirements.txt    # Python dependencies
 ┣ 📄 params.yaml         # Centralized parameters
 ┣ 📄 dvc.yaml            # DVC pipeline definition
 ┣ 📄 MLproject           # MLflow project definition
 ┣ 📄 README.md           # Project documentation (this file)
```

---

## Key Features

- **EDA and Data Cleaning**:  Thorough exploration and cleaning of ~1.3M transaction records.
- **Data Versioning**: Using **DVC** to track versions of raw, cleaned, and model-ready datasets.
- **Handling Imbalanced Data**: Techniques like **RandomUnderSampler** and **SMOTETomek** to balance classes.
- **Feature Engineering**: Time-based aggregations, categorical encoding, and lagged feature generation.
- **Modeling**: Trained multiple classifiers (Logistic Regression, KNN, SVC, Decision Tree) with hyperparameter tuning.
- **Experiment Tracking**: Logged experiments with **MLflow**, tracking accuracy and ROC-AUC.
- **Visualization**: Interactive Plotly plots for fraud distribution, transaction trends, and model performance.

---

## EDA & Insights

- **Fraud distribution**: Extremely imbalanced dataset (~0.5% fraud).
- **Amount Analysis**: Fraudulent transactions often have smaller average amounts.
- **Category Analysis**: Certain merchant categories have higher fraud rates.
- **Time Series Decomposition**: Decomposed transaction volumes into seasonal, trend, and residual components.
- **Outlier Detection**: Applied IQR and Z-score methods for amount outlier analysis.

---

## Machine Learning Pipeline

| Step | Description |
|:--|:--|
| 1. Data Cleaning       | Drop missing values, fix data types, engineer features |
| 2. Data Versioning     | Track dataset versions with DVC |
| 3. Handling Imbalance  | Apply undersampling and SMOTETomek |
| 4. Model Training      | Train classifiers: LR, KNN, SVC, DT |
| 5. Hyperparameter Tuning | Use GridSearchCV for optimization |
| 6. Model Evaluation    | Generate classification reports and ROC curves |
| 7. Experiment Tracking | Log runs in MLflow |

---

## Model Evaluation (Best Models)

| Model                     | Accuracy | ROC-AUC |
|:------------------------- |:--------:|:-------:|
| Logistic Regression       | XX%      | XX      |
| K-Nearest Neighbors       | XX%      | XX      |
| Support Vector Classifier | XX%      | XX      |
| Decision Tree             | XX%      | XX      |

*(Actual metrics will vary based on your experiments.)*

---

## MLOps Stack

| Tool            | Purpose                                  |
|:--------------- |:----------------------------------------|
| **DVC**         | Data versioning and pipeline management  |
| **MLflow**      | Experiment tracking and model registry   |
| **Git**         | Code version control                     |
| **Scikit-learn**| Modeling and preprocessing               |
| **Plotly**      | Interactive visualizations               |
| **Pandas/Numpy**| Data manipulation                        |
| **imbalanced-learn** | Class balancing techniques           |

---

## Installation

```bash
# Clone repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Install dependencies
pip install -r requirements.txt
```

---

## Future Improvements

- Add deep learning models (TCN, LSTM)
- Build ensemble meta-model
- Create Streamlit UI for real-time prediction
- Deploy with Docker on AWS/GCP
- Integrate CI/CD for full MLOps pipeline

---

## Acknowledgments

- [Credit Card Transactions Dataset - Kaggle](https://www.kaggle.com/...) 
- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)

---

## Contact

Feel free to connect:

- [LinkedIn](https://www.linkedin.com/in/shyamal-bhatt/)
- Email: <bhattshyamal478@gmail.com>

