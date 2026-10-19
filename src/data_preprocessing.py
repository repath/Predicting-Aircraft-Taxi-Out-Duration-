"""
data_preprocessing.py
Implements Step 2: Data Preprocessing & Cleaning for the Taxi-Out Prediction project.

Guidelines:
1. Filter Out Invalid Flights:
   - Remove cancelled flights (CANCELLED == 1)
   - Remove diverted flights (DIVERTED == 1)
   - Filter out missing or non-positive TAXI_OUT values (<= 0)
2. Outlier Removal & Capping:
   - Analyze percentile distribution (50th, 90th, 95th, 99th, 99.5th, 99.9th)
   - Filter or cap extreme anomalies above 90-120 minutes (e.g. severe ground stops/de-icing)
"""

import os
import json
import pandas as pd
import numpy as np

def inspect_outliers(series, percentiles=[50, 75, 90, 95, 99, 99.5, 99.9]):
    """Calculate and return detailed percentile statistics for outlier inspection."""
    pct_dict = {}
    for p in percentiles:
        pct_dict[f"p{p}"] = float(np.percentile(series, p))
    pct_dict["min"] = float(series.min())
    pct_dict["max"] = float(series.max())
    pct_dict["mean"] = float(series.mean())
    pct_dict["std"] = float(series.std())
    pct_dict["count_above_90"] = int((series > 90).sum())
    pct_dict["count_above_120"] = int((series > 120).sum())
    return pct_dict

def clean_flight_data(
    input_path="data/raw/bts_carrier_ontime_sample.csv",
    output_path="data/processed/cleaned_flights.csv",
    outlier_strategy="filter", # 'filter' or 'cap'
    outlier_threshold=120.0
):
    """
    Cleans raw BTS flight dataset according to project guidelines.
    Returns cleaned DataFrame and audit log.
    """
    print(f"Loading raw data from: {input_path}")
    df_raw = pd.read_csv(input_path)
    total_raw = len(df_raw)
    
    # Audit trail tracking
    audit_log = {
        "raw_record_count": total_raw,
        "initial_columns": df_raw.columns.tolist()
    }
    
    # Required core features to verify
    required_cols = [
        'FL_DATE', 'OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST',
        'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'DISTANCE',
        'TAXI_OUT', 'CANCELLED', 'DIVERTED'
    ]
    missing_required = [c for c in required_cols if c not in df_raw.columns]
    if missing_required:
        raise ValueError(f"Dataset missing required columns: {missing_required}")
    
    # Step 2.1: Filter out cancelled flights
    cancelled_count = int((df_raw['CANCELLED'] == 1).sum())
    df_step1 = df_raw[df_raw['CANCELLED'] == 0].copy()
    audit_log["cancelled_flights_removed"] = cancelled_count
    
    # Step 2.2: Filter out diverted flights
    diverted_count = int((df_step1['DIVERTED'] == 1).sum())
    df_step2 = df_step1[df_step1['DIVERTED'] == 0].copy()
    audit_log["diverted_flights_removed"] = diverted_count
    
    # Step 2.3: Filter missing or negative/zero TAXI_OUT
    invalid_taxi_mask = df_step2['TAXI_OUT'].isna() | (df_step2['TAXI_OUT'] <= 0)
    invalid_taxi_count = int(invalid_taxi_mask.sum())
    df_step3 = df_step2[~invalid_taxi_mask].copy()
    audit_log["invalid_or_missing_taxi_out_removed"] = invalid_taxi_count
    
    # Also clean missing DEP_TIME or CRS_DEP_TIME if any
    missing_dep_time = int(df_step3['DEP_TIME'].isna().sum())
    if missing_dep_time > 0:
        df_step3 = df_step3.dropna(subset=['DEP_TIME', 'CRS_DEP_TIME'])
        audit_log["missing_dep_time_removed"] = missing_dep_time
    
    # Step 2.4: Outlier Percentile Inspection
    taxi_series = df_step3['TAXI_OUT']
    percentile_stats = inspect_outliers(taxi_series)
    audit_log["percentile_inspection_before_outlier_treatment"] = percentile_stats
    
    # Outlier treatment (filter or cap based on threshold)
    if outlier_strategy == "filter":
        outlier_mask = df_step3['TAXI_OUT'] > outlier_threshold
        outliers_count = int(outlier_mask.sum())
        df_cleaned = df_step3[~outlier_mask].copy()
        audit_log["outlier_strategy"] = "filter"
        audit_log["outlier_threshold"] = outlier_threshold
        audit_log["outliers_removed"] = outliers_count
    elif outlier_strategy == "cap":
        outliers_count = int((df_step3['TAXI_OUT'] > outlier_threshold).sum())
        df_cleaned = df_step3.copy()
        df_cleaned['TAXI_OUT'] = np.clip(df_cleaned['TAXI_OUT'], a_min=None, a_max=outlier_threshold)
        audit_log["outlier_strategy"] = "cap"
        audit_log["outlier_threshold"] = outlier_threshold
        audit_log["outliers_capped"] = outliers_count
    else:
        df_cleaned = df_step3.copy()
        audit_log["outlier_strategy"] = "none"
    
    audit_log["final_cleaned_count"] = len(df_cleaned)
    audit_log["retention_rate_pct"] = round((len(df_cleaned) / total_raw) * 100, 2)
    audit_log["post_clean_stats"] = {
        "mean_taxi_out": float(df_cleaned['TAXI_OUT'].mean()),
        "std_taxi_out": float(df_cleaned['TAXI_OUT'].std()),
        "min_taxi_out": float(df_cleaned['TAXI_OUT'].min()),
        "max_taxi_out": float(df_cleaned['TAXI_OUT'].max()),
        "median_taxi_out": float(df_cleaned['TAXI_OUT'].median())
    }
    
    # Ensure date column is formatted properly
    df_cleaned['FL_DATE'] = pd.to_datetime(df_cleaned['FL_DATE']).dt.strftime('%Y-%m-%d')
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_cleaned.to_csv(output_path, index=False)
    
    # Save audit summary
    audit_summary_path = os.path.join(os.path.dirname(output_path), "cleaning_audit_summary.json")
    with open(audit_summary_path, "w") as f:
        json.dump(audit_log, f, indent=2)
        
    print(f"Data cleaning complete! Cleaned dataset saved to: {output_path}")
    print(f"Initial: {total_raw:,} | Cleaned: {len(df_cleaned):,} ({audit_log['retention_rate_pct']}%)")
    print(f"Taxi-Out summary: Mean={audit_log['post_clean_stats']['mean_taxi_out']:.2f} min, "
          f"Median={audit_log['post_clean_stats']['median_taxi_out']:.1f} min, "
          f"Max={audit_log['post_clean_stats']['max_taxi_out']:.1f} min")
    print(f"Audit log saved to: {audit_summary_path}")
    
    return df_cleaned, audit_log

if __name__ == "__main__":
    clean_flight_data()
