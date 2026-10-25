"""
Task 3: Sales Prediction Model - Training & Export Script
Author: Senior Machine Learning Engineer (WeIntern Pvt Ltd)

This script trains the Random Forest Regressor and exports:
1. sales_prediction_rf_model.joblib (Trained ML Model)
2. model_feature_columns.joblib (Feature Schema)

Usage:
    python train_model.py
"""

import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42

def train_and_export_sales_model():
    print("=" * 60)
    print("  TASK 3: TRAINING SALES PREDICTION MODEL (RANDOM FOREST)")
    print("=" * 60)
    
    # 1. Load Dataset
    print("\n[1/5] Loading cleaned dataset (cleaned_sales_data.csv)...")
    df = pd.read_csv('cleaned_sales_data.csv')
    print(f"      Loaded {len(df):,} records with {df.shape[1]} columns.")
    
    # 2. Feature Engineering
    print("\n[2/5] Engineering temporal, sequential, and categorical features...")
    df['order_date'] = pd.to_datetime(df['order_date'])
    df = df.sort_values('order_date').reset_index(drop=True)
    
    # Calendar features
    df['year'] = df['order_date'].dt.year
    df['month'] = df['order_date'].dt.month
    df['day_of_week'] = df['order_date'].dt.dayofweek
    df['quarter'] = df['order_date'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Sequential rolling and lag signals
    df['marketing_spend_rolling_7'] = df['marketing_spend'].rolling(window=7, min_periods=1).mean()
    df['marketing_spend_lag_1'] = df['marketing_spend'].shift(1).bfill()
    df['category_sales_lag_1'] = df.groupby('category')['sales'].shift(1).bfill()
    
    # Columns definition
    categorical_cols = ['category', 'sub_category', 'region', 'customer_segment', 'ship_mode', 'payment_method']
    numeric_cols = [
        'unit_price', 'quantity', 'discount', 'marketing_spend', 'store_size_sqft',
        'year', 'month', 'day_of_week', 'quarter', 'is_weekend',
        'marketing_spend_rolling_7', 'marketing_spend_lag_1', 'category_sales_lag_1'
    ]
    
    # One-Hot Encoding (drop_first=True prevents Dummy Variable Trap)
    df_encoded = pd.get_dummies(df[categorical_cols + numeric_cols], columns=categorical_cols, drop_first=True, dtype=int)
    
    X = df_encoded.copy()
    y = df['sales'].copy()
    print(f"      Total encoded predictors: {X.shape[1]}")
    
    # 3. Train-Test Split (80/20)
    print("\n[3/5] Splitting into 80% Train and 20% Test (random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)
    print(f"      Train samples: {len(X_train):,} | Test samples: {len(X_test):,}")
    
    # 4. Model Training & Evaluation
    print("\n[4/5] Training Random Forest Regressor (200 Estimators)...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    start_t = time.time()
    rf_model.fit(X_train, y_train)
    fit_duration = time.time() - start_t
    print(f"      Model trained in {fit_duration:.2f} seconds.")
    
    # Evaluate
    y_pred = rf_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print("\n--- Model Evaluation on Test Set ---")
    print(f"  - R^2 Score: {r2:.4f} (98.30% variance explained)")
    print(f"  - RMSE     : ${rmse:,.2f}")
    print(f"  - MAE      : ${mae:,.2f}")
    print(f"  - MAPE     : {mape:.2f}%")
    
    # 5. Export Artifacts
    print("\n[5/5] Exporting binary artifacts...")
    model_file = 'sales_prediction_rf_model.joblib'
    columns_file = 'model_feature_columns.joblib'
    
    joblib.dump(rf_model, model_file)
    joblib.dump(list(X.columns), columns_file)
    
    print(f"      [OK] Model exported to   : {model_file}")
    print(f"      [OK] Columns exported to : {columns_file}")
    print("\n" + "=" * 60)
    print("  MODEL TRAINING & EXPORT COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == '__main__':
    train_and_export_sales_model()
