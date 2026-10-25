# WeIntern Pvt Ltd | Task 3: Sales Prediction Model

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.0-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-3.0.6-darkblue.svg)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, production-grade supervised machine learning pipeline designed to predict transaction-level sales revenue using historical retail and enterprise data. Developed for **Task 3: Sales Prediction Model** as part of the WeIntern Pvt Ltd Data Science & Machine Learning Internship handbook.

---

## 📌 Project Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        END-TO-END PREDICTIVE MODELING PIPELINE                         │
├─────────────────────┬──────────────────────┬─────────────────────┬─────────────────────┤
│ 1. Data Ingestion   │ 2. Feature Eng.      │ 3. Model Training   │ 4. Diagnostics &    │
│  - Load Task 1 CSV  │  - Datetime Signals  │  - OLS Baseline     │    Inference        │
│  - Target Definition│  - Rolling & Lags    │  - Random Forest    │  - GridSearch Tuning│
│  - ID Exclusion     │  - One-Hot Encoding  │  - Gradient Boost   │  - Residual Norm    │
│  - Skew Analysis    │  - StandardScaler    │  - Leakage Guard    │  - Joblib Export    │
└─────────────────────┴──────────────────────┴─────────────────────┴─────────────────────┘
```

---

## 📊 Dataset Overview

- **Source File:** `cleaned_sales_data.csv` (Task 1 data wrangling artifact)
- **Observations:** 2,500 transaction records
- **Temporal Span:** January 1, 2023 – December 31, 2024 (2 full calendar years)
- **Features:** 14 predictors across product pricing, category hierarchies, geographic regions, customer segments, fulfillment modes, and marketing expenditure.
- **Target Variable ($y$):** Continuous sales value (`sales`):
  - **Mean:** \$1,244.86
  - **Median:** \$742.66
  - **Std Dev:** \$1,290.60
  - **Range:** [\$23.07, \$7,385.23]
  - **Distribution:** Moderately right-skewed (+1.77 skewness)

---

## ⚙️ Feature Engineering & Preprocessing

1. **Temporal Signal Extraction:**
   - Deconstructed `order_date` into `year`, `month`, `day_of_week`, `quarter`, and `is_weekend`.
2. **Sequential Rolling Windows & Lags:**
   - 7-period rolling marketing expenditure (`marketing_spend_rolling_7`).
   - 1-step previous transaction spend (`marketing_spend_lag_1`).
   - Category-specific lagged sales (`category_sales_lag_1`) strictly shifted to prevent lookahead data leakage.
3. **Categorical Encoding:**
   - One-Hot Encoding (`pd.get_dummies(..., drop_first=True, dtype=int)`) on nominal variables (`category`, `sub_category`, `region`, `customer_segment`, `ship_mode`, `payment_method`) to prevent multicollinearity and the Dummy Variable Trap.
4. **Predictor Scaling & Leakage Guardrails:**
   - Continuous variables normalized using `StandardScaler`.
   - Scaler fitted strictly on the 80% training partition ($N = 2,000$) and transformed across the 20% test partition ($N = 500$).

---

## 🏆 Model Benchmark & Evaluation Results

Evaluated on the held-out test partition ($N = 500$ unseen transactions, `random_state=42`):

| Model Architecture | RMSE ($) | MAE ($) | $R^2$ Score | MAPE (%) | Key Insights |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Linear Regression (Baseline)** | \$505.48 | \$352.92 | 0.8569 | 77.42% | Misses non-linear pricing $\times$ quantity interactions |
| **Random Forest Regressor (Ensemble)** | **\$177.01** | **\$96.56** | **0.9825** | **8.99%** | **65.0% RMSE reduction** over OLS baseline |
| **Gradient Boosting (Benchmark)** | \$98.72 | \$62.88 | 0.9945 | 8.03% | Sequential residual shrinkage minimization |
| **Tuned Random Forest (GridSearchCV)** | **\$176.60** | **\$97.12** | **0.9825** | **8.96%** | Optimized via 3-Fold Cross-Validation |

---

## 🔍 Model Diagnostics & Interpretability

- **Top Predictors:** `unit_price` (65.7%) and `quantity` (32.5%) account for over 98% of total predictive power, followed by `discount`, `marketing_spend`, and seasonal month signals.
- **Actual vs. Predicted Plot:** Strong linear clustering along the $y = x$ parity diagonal across both retail transactions and enterprise bulk orders.
- **Residual Distribution:** Residual errors are centered symmetrically at zero ($\mu_{error} \approx -\$0.72$) with homoscedastic variance.

---

## 🚀 Quickstart & Reproduction

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/sales-prediction-model.git
cd sales-prediction-model
```

### 2. Install Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter joblib
```

### 3. Launch the Jupyter Notebook
```bash
jupyter notebook 03_sales_prediction_model.ipynb
```

### 4. Run Production Inference
```python
import joblib
import pandas as pd

# Load serialized model
model = joblib.load('sales_prediction_rf_model.joblib')
features = joblib.load('model_feature_columns.joblib')
```

---

## 📂 Repository Structure

```
sales-prediction-model/
├── 03_sales_prediction_model.ipynb   # Complete, executed production notebook
├── cleaned_sales_data.csv            # Cleaned input dataset
├── sales_prediction_rf_model.joblib  # Serialized production Random Forest model
├── model_feature_columns.joblib      # Expected feature column schema
├── .gitignore                        # Git exclusion rules
└── README.md                         # Project documentation
```

---

## 📄 License
This project is developed under the MIT License for educational and technical demonstration purposes.
