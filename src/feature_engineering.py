"""
feature_engineering.py
Implements Step 3: Feature Engineering for the Taxi-Out Prediction project.

Guidelines:
1. Temporal Features:
   - HOUR: Extract departure hour from CRS_DEP_TIME.
   - DAY_OF_WEEK, MONTH.
   - Cyclical encoding for time: Sin_Hour = sin(2*pi*Hour/24), Cos_Hour = cos(2*pi*Hour/24).
2. Airport Congestion Index:
   - Rolling count of concurrent departures from the same ORIGIN within 30-min and 1-hour window.
3. Target/Categorical Encoding:
   - Mean taxi-out time per ORIGIN airport and per OP_UNIQUE_CARRIER.
   - Strictly computed on the training set (or with smoothing) to prevent target leakage.
4. Time-Based Train/Test Split:
   - Train on Jan-Oct (Months 1-10), Test on Nov-Dec (Months 11-12).
"""

import os
import json
import pandas as pd
import numpy as np

def extract_temporal_features(df):
    """Extracts Hour, Day of Week, Month, and cyclical sin/cos encodings."""
    df = df.copy()
    
    # 1. Hour extraction from CRS_DEP_TIME (HHMM format)
    df['HOUR'] = (df['CRS_DEP_TIME'] // 100).astype(int)
    # Clamp in 0..23 in case of 2400
    df['HOUR'] = df['HOUR'].clip(0, 23)
    
    # 2. Date features
    fl_date = pd.to_datetime(df['FL_DATE'])
    df['DAY_OF_WEEK'] = fl_date.dt.dayofweek # 0=Monday, 6=Sunday
    df['MONTH'] = fl_date.dt.month # 1..12
    
    # 3. Cyclical hour encoding
    df['SIN_HOUR'] = np.sin(2 * np.pi * df['HOUR'] / 24.0)
    df['COS_HOUR'] = np.cos(2 * np.pi * df['HOUR'] / 24.0)
    
    return df

def compute_airport_congestion(df, windows=['60min', '30min']):
    """
    Computes rolling departure congestion index per ORIGIN airport.
    Calculates number of departing flights within preceding 60-min and 30-min windows.
    """
    df = df.copy()
    
    # Construct exact departure datetime for sequencing
    dep_h = (df['DEP_TIME'].astype(int) // 100).clip(0, 23)
    dep_m = (df['DEP_TIME'].astype(int) % 100).clip(0, 59)
    df['dep_datetime'] = pd.to_datetime(df['FL_DATE']) + pd.to_timedelta(dep_h, unit='h') + pd.to_timedelta(dep_m, unit='m')
    
    # Sort chronologically by origin and departure time
    df = df.sort_values(['ORIGIN', 'dep_datetime']).reset_index(drop=True)
    
    dfs = []
    for orig, grp in df.groupby('ORIGIN'):
        grp = grp.sort_values('dep_datetime').copy()
        indexed = grp.set_index('dep_datetime')
        c_60m = indexed['TAXI_OUT'].rolling('60min', closed='both').count() - 1
        c_30m = indexed['TAXI_OUT'].rolling('30min', closed='both').count() - 1
        grp['CONGESTION_1H'] = np.maximum(0, c_60m.values)
        grp['CONGESTION_30M'] = np.maximum(0, c_30m.values)
        dfs.append(grp)
    grouped = pd.concat(dfs, axis=0)
    grouped = grouped.sort_values('dep_datetime').reset_index(drop=True)
    return grouped

def apply_target_encoding(train_df, test_df, smoothing_weight=10.0):
    """
    Computes mean TAXI_OUT target encoding for ORIGIN and OP_UNIQUE_CARRIER
    using ONLY the training data with empirical Bayes smoothing.
    Maps encoded values to both train and test sets to guarantee zero leakage.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    global_mean = float(train_df['TAXI_OUT'].mean())
    
    enc_maps = {
        "global_mean": global_mean,
        "origin_encoding": {},
        "carrier_encoding": {}
    }
    
    # 1. Target encoding for ORIGIN
    orig_stats = train_df.groupby('ORIGIN')['TAXI_OUT'].agg(['count', 'mean'])
    for orig, row in orig_stats.iterrows():
        count, mean_val = row['count'], row['mean']
        smoothed = (count * mean_val + smoothing_weight * global_mean) / (count + smoothing_weight)
        enc_maps["origin_encoding"][orig] = float(smoothed)
        
    train_df['ORIGIN_TAXI_MEAN'] = train_df['ORIGIN'].map(enc_maps["origin_encoding"]).fillna(global_mean)
    test_df['ORIGIN_TAXI_MEAN'] = test_df['ORIGIN'].map(enc_maps["origin_encoding"]).fillna(global_mean)
    
    # 2. Target encoding for OP_UNIQUE_CARRIER
    carrier_stats = train_df.groupby('OP_UNIQUE_CARRIER')['TAXI_OUT'].agg(['count', 'mean'])
    for carrier, row in carrier_stats.iterrows():
        count, mean_val = row['count'], row['mean']
        smoothed = (count * mean_val + smoothing_weight * global_mean) / (count + smoothing_weight)
        enc_maps["carrier_encoding"][carrier] = float(smoothed)
        
    train_df['CARRIER_TAXI_MEAN'] = train_df['OP_UNIQUE_CARRIER'].map(enc_maps["carrier_encoding"]).fillna(global_mean)
    test_df['CARRIER_TAXI_MEAN'] = test_df['OP_UNIQUE_CARRIER'].map(enc_maps["carrier_encoding"]).fillna(global_mean)
    
    return train_df, test_df, enc_maps

def run_feature_engineering(
    cleaned_input_path="data/processed/cleaned_flights.csv",
    train_output_path="data/processed/train_features.csv",
    test_output_path="data/processed/test_features.csv",
    split_month_threshold=10 # Months 1-10 Train (Jan-Oct), Months 11-12 Test (Nov-Dec)
):
    """
    Executes full Step 3 feature engineering pipeline.
    """
    print(f"Loading cleaned dataset from: {cleaned_input_path}")
    df = pd.read_csv(cleaned_input_path)
    
    # 1. Temporal & Cyclical Features
    print("Extracting temporal features (Hour, Day of Week, Month, Cyclical Sin/Cos)...")
    df = extract_temporal_features(df)
    
    # 2. Airport Congestion Index
    print("Calculating Airport Congestion Index (30-minute and 1-hour rolling departures)...")
    df = compute_airport_congestion(df)
    
    # 3. Time-Based Train/Test Split (Jan-Oct Train, Nov-Dec Test)
    print(f"Performing time-based split: Train (Months 1-{split_month_threshold}), Test (Months {split_month_threshold+1}-12)...")
    train_mask = df['MONTH'] <= split_month_threshold
    train_df = df[train_mask].copy().reset_index(drop=True)
    test_df = df[~train_mask].copy().reset_index(drop=True)
    
    print(f"Train set records: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Test set records:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    
    # 4. Target Encoding strictly on Train set
    print("Applying target encoding on ORIGIN and OP_UNIQUE_CARRIER using training statistics...")
    train_df, test_df, enc_maps = apply_target_encoding(train_df, test_df)
    
    # Save datasets
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    
    # Save target encoding lookup tables for deployment/inference
    enc_path = "models/target_encoders.json"
    os.makedirs(os.path.dirname(enc_path), exist_ok=True)
    with open(enc_path, "w") as f:
        json.dump(enc_maps, f, indent=2)
        
    print(f"Engineered features successfully saved:")
    print(f" - Train: {train_output_path}")
    print(f" - Test:  {test_output_path}")
    print(f" - Target encodings: {enc_path}")
    
    return train_df, test_df, enc_maps

if __name__ == "__main__":
    run_feature_engineering()
