"""
Sales Prediction Model - Standalone CLI Inference Tool
Usage:
    python predict.py
"""

import joblib
import pandas as pd
import numpy as np

def load_sales_model(model_path='sales_prediction_rf_model.joblib', features_path='model_feature_columns.joblib'):
    """Load the serialized Random Forest model and column schema."""
    print("Loading production model artifact...")
    model = joblib.load(model_path)
    feature_columns = joblib.load(features_path)
    print(f"[OK] Model successfully loaded: {type(model).__name__}")
    print(f"[OK] Ensemble configuration : {len(model.estimators_)} Decision Trees")
    print(f"[OK] Expected input features: {len(feature_columns)} columns\n")
    return model, feature_columns

def predict_single_order(model, feature_columns, order_details):
    """Generate a predicted sales figure for a single order payload."""
    df_raw = pd.DataFrame([order_details])
    df_raw['order_date'] = pd.to_datetime(df_raw['order_date'])
    
    # Calendar signal extraction
    df_raw['year'] = df_raw['order_date'].dt.year
    df_raw['month'] = df_raw['order_date'].dt.month
    df_raw['day_of_week'] = df_raw['order_date'].dt.dayofweek
    df_raw['quarter'] = df_raw['order_date'].dt.quarter
    df_raw['is_weekend'] = df_raw['day_of_week'].isin([5, 6]).astype(int)
    
    # Fill rolling / lag defaults if not explicitly provided
    if 'marketing_spend_rolling_7' not in df_raw:
        df_raw['marketing_spend_rolling_7'] = df_raw['marketing_spend']
    if 'marketing_spend_lag_1' not in df_raw:
        df_raw['marketing_spend_lag_1'] = df_raw['marketing_spend']
    if 'category_sales_lag_1' not in df_raw:
        df_raw['category_sales_lag_1'] = df_raw['unit_price'] * df_raw['quantity']
        
    categorical_cols = ['category', 'sub_category', 'region', 'customer_segment', 'ship_mode', 'payment_method']
    numeric_cols = [
        'unit_price', 'quantity', 'discount', 'marketing_spend', 'store_size_sqft',
        'year', 'month', 'day_of_week', 'quarter', 'is_weekend',
        'marketing_spend_rolling_7', 'marketing_spend_lag_1', 'category_sales_lag_1'
    ]
    
    # Encode categorical dummy variables
    encoded = pd.get_dummies(df_raw[categorical_cols + numeric_cols], columns=categorical_cols, drop_first=True, dtype=int)
    
    # Reindex columns to match exact training matrix
    aligned = encoded.reindex(columns=feature_columns, fill_value=0)
    
    prediction = model.predict(aligned)[0]
    return float(prediction)

if __name__ == '__main__':
    model, features = load_sales_model()
    
    # Demonstration Test Cases
    sample_orders = [
        {
            'order_date': '2025-06-15',
            'category': 'Technology',
            'sub_category': 'Printers',
            'region': 'South',
            'customer_segment': 'Corporate',
            'ship_mode': 'Standard Class',
            'payment_method': 'Credit Card',
            'unit_price': 350.00,
            'quantity': 5,
            'discount': 0.05,
            'marketing_spend': 250.00,
            'store_size_sqft': 22000
        },
        {
            'order_date': '2025-06-15',
            'category': 'Office Supplies',
            'sub_category': 'Paper',
            'region': 'East',
            'customer_segment': 'Consumer',
            'ship_mode': 'First Class',
            'payment_method': 'E-Wallet',
            'unit_price': 25.00,
            'quantity': 3,
            'discount': 0.00,
            'marketing_spend': 80.00,
            'store_size_sqft': 15000
        }
    ]
    
    print("=== Model Inference Demonstration ===")
    for i, order in enumerate(sample_orders, 1):
        pred = predict_single_order(model, features, order)
        print(f"\nOrder #{i} [{order['category']} - {order['sub_category']}]:")
        print(f"  - Unit Price    : ${order['unit_price']:,.2f}")
        print(f"  - Quantity      : {order['quantity']}")
        print(f"  - Discount      : {order['discount'] * 100:.0f}%")
        print(f"  - Predicted Revenue: ${pred:,.2f}")
