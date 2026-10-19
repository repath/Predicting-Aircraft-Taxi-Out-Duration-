"""
generate_sample_data.py
Generates a realistic BTS Marketing Carrier On-Time Performance dataset
spanning January 1, 2025 to December 31, 2025 (focusing on DFW hub operations
and peer major US hubs).
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_bts_data(num_records=75000, random_seed=42, output_path="data/raw/bts_carrier_ontime_sample.csv"):
    np.random.seed(random_seed)
    print(f"Generating {num_records} realistic BTS flight records for 2025...")
    
    # 1. Date range: Full year 2025 (Jan 1 to Dec 31)
    start_date = datetime(2025, 1, 1)
    day_offsets = np.random.randint(0, 365, size=num_records)
    fl_dates = [start_date + timedelta(days=int(d)) for d in day_offsets]
    fl_date_strs = [d.strftime("%Y-%m-%d") for d in fl_dates]
    
    # 2. Origin airports: DFW focused (~70%), with peer major hubs
    origins = np.random.choice(
        ['DFW', 'ATL', 'ORD', 'DEN', 'LAX', 'CLT', 'JFK'],
        size=num_records,
        p=[0.70, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03]
    )
    
    # 3. Destination airports
    dest_pool = ['ORD', 'ATL', 'LAX', 'LGA', 'MIA', 'DEN', 'SFO', 'PHX', 'IAH', 'BOS', 'SEA', 'LAS', 'MCO', 'EWR', 'DTW', 'MSP']
    dests = []
    for orig in origins:
        choices = [d for d in dest_pool if d != orig]
        dests.append(np.random.choice(choices))
    
    # 4. Carriers: At DFW, American (AA) is dominant; others operate hub feeder / spoke
    carriers = []
    for orig in origins:
        if orig == 'DFW':
            c = np.random.choice(['AA', 'WN', 'DL', 'UA', 'NK', 'B6'], p=[0.72, 0.12, 0.06, 0.05, 0.03, 0.02])
        elif orig == 'ATL':
            c = np.random.choice(['DL', 'WN', 'AA', 'UA', 'NK'], p=[0.75, 0.10, 0.06, 0.06, 0.03])
        elif orig == 'ORD':
            c = np.random.choice(['UA', 'AA', 'WN', 'DL', 'NK'], p=[0.45, 0.40, 0.07, 0.05, 0.03])
        else:
            c = np.random.choice(['AA', 'DL', 'UA', 'WN', 'AS', 'B6'], p=[0.30, 0.25, 0.25, 0.10, 0.05, 0.05])
        carriers.append(c)
    
    # 5. Scheduled Departure Time (CRS_DEP_TIME): realistic aviation banking distribution
    # Peak departure waves: 7:00-9:00, 11:00-13:30, 16:00-19:30
    hours_dist = np.random.choice(
        range(5, 24),
        size=num_records,
        p=[0.02, 0.04, 0.07, 0.08, 0.08, 0.06, 0.07, 0.08, 0.06, 0.06, 0.07, 0.08, 0.08, 0.06, 0.04, 0.02, 0.01, 0.01, 0.01]
    )
    minutes_dist = np.random.choice(range(0, 60, 5), size=num_records)
    crs_dep_times = hours_dist * 100 + minutes_dist
    
    # 6. Departure Delays (DEP_DELAY): Exponential tail with on-time bias
    # ~75% on-time (-5 to +10 mins), 25% delayed (>15 mins)
    is_delayed = np.random.rand(num_records) < 0.25
    dep_delays = np.where(
        is_delayed,
        np.random.exponential(scale=35.0, size=num_records) + 15,
        np.random.normal(loc=-2.0, scale=6.0, size=num_records)
    ).round(0)
    
    # Compute actual DEP_TIME (HHMM format)
    dep_times = []
    for crs, delay in zip(crs_dep_times, dep_delays):
        h = crs // 100
        m = crs % 100
        total_mins = h * 60 + m + int(delay)
        total_mins = max(0, total_mins % 1440) # 24h wrap
        dep_times.append((total_mins // 60) * 100 + (total_mins % 60))
    dep_times = np.array(dep_times)
    
    # 7. Distance (miles): hub-and-spoke flights
    distances = np.random.choice(
        [300, 450, 700, 950, 1200, 1600, 2100, 2500],
        size=num_records,
        p=[0.15, 0.20, 0.25, 0.15, 0.10, 0.08, 0.05, 0.02]
    ) + np.random.randint(-40, 40, size=num_records)
    
    # 8. Flight Status: Cancellations (~2.0%) and Diversions (~0.3%)
    cancelled_flags = (np.random.rand(num_records) < 0.020).astype(int)
    diverted_flags = (np.random.rand(num_records) < 0.003).astype(int)
    # Ensure cancelled flights are not also marked diverted
    diverted_flags = np.where(cancelled_flags == 1, 0, diverted_flags)
    
    # 9. Realistic Taxi-Out Calculation (Physics & Operations of Airport Surface)
    # Baseline airport layout complexity
    airport_base_taxi = {
        'DFW': 17.5, # Sprawling 7 runways, large taxi distances
        'ORD': 19.0, # High congestion layout
        'ATL': 16.0, # Parallel runways, efficient layout
        'DEN': 16.8, # Huge landmass, long taxiways
        'LAX': 17.2, # Dense apron, runway crossings
        'CLT': 18.0, # High volume single alleyways
        'JFK': 22.0  # Intense congestion, metering delays
    }
    
    base_taxi = np.array([airport_base_taxi.get(orig, 16.5) for orig in origins])
    
    # Hour effect: Morning and Evening rush hours increase surface queueing
    hour_surge = np.where(
        (hours_dist >= 7) & (hours_dist <= 9), 4.5,
        np.where((hours_dist >= 16) & (hours_dist <= 19), 5.5,
        np.where((hours_dist >= 11) & (hours_dist <= 14), 2.5, 0.0))
    )
    
    # Weather / Seasonality: Summer thunderstorms (Jun-Aug) and winter de-icing (Jan-Feb)
    months = np.array([d.month for d in fl_dates])
    season_surge = np.where(
        (months >= 6) & (months <= 8), 1.8, # summer convective activity
        np.where((months == 1) | (months == 2) | (months == 12), 2.2, 0.0) # winter de-icing
    )
    
    # Carrier effect (hub carriers maintain higher peak bank queueing)
    carrier_surge = np.where(
        (np.array(carriers) == 'AA') & (origins == 'DFW'), 2.0,
        np.where((np.array(carriers) == 'DL') & (origins == 'ATL'), 1.8,
        np.where((np.array(carriers) == 'UA') & (origins == 'ORD'), 1.5, 0.0))
    )
    
    # Distance / Aircraft size effect (longer distance flights use longer runways)
    distance_effect = (distances / 1000.0) * 0.8
    
    # Random queuing delay (Gamma distribution simulates right-skewed queuing)
    queuing_noise = np.random.gamma(shape=2.5, scale=2.0, size=num_records) - 5.0
    
    # Compute base TAXI_OUT
    simulated_taxi = base_taxi + hour_surge + season_surge + carrier_surge + distance_effect + queuing_noise
    simulated_taxi = np.clip(simulated_taxi, 5.0, 95.0).round(1)
    
    # Inject deliberate dirty / outlier data to fulfill Step 2 preprocessing requirements:
    # A) Outliers (>120 minutes) representing de-icing ground stops (~0.8%)
    outlier_idx = np.random.choice(num_records, size=int(num_records * 0.008), replace=False)
    simulated_taxi[outlier_idx] = np.random.uniform(121.0, 175.0, size=len(outlier_idx)).round(1)
    
    # B) Borderline high taxi-out (90-120 minutes) (~1.2%)
    borderline_idx = np.random.choice(
        [i for i in range(num_records) if i not in outlier_idx],
        size=int(num_records * 0.012),
        replace=False
    )
    simulated_taxi[borderline_idx] = np.random.uniform(91.0, 119.0, size=len(borderline_idx)).round(1)
    
    # C) Missing / Negative taxi-out values (~0.6%)
    dirty_val_idx = np.random.choice(
        [i for i in range(num_records) if i not in outlier_idx and i not in borderline_idx],
        size=int(num_records * 0.006),
        replace=False
    )
    for idx in dirty_val_idx[:len(dirty_val_idx)//2]:
        simulated_taxi[idx] = np.nan
    for idx in dirty_val_idx[len(dirty_val_idx)//2:]:
        simulated_taxi[idx] = np.random.choice([-1.0, -10.0, -4.0])
    
    # D) Cancelled flights have missing TAXI_OUT and missing DEP_TIME
    simulated_taxi = np.where(cancelled_flags == 1, np.nan, simulated_taxi)
    dep_times = np.where(cancelled_flags == 1, np.nan, dep_times)
    dep_delays = np.where(cancelled_flags == 1, np.nan, dep_delays)
    
    # 10. Wheels Off Time (WHEELS_OFF): DEP_TIME + TAXI_OUT
    wheels_off = []
    for dt, tx, canc in zip(dep_times, simulated_taxi, cancelled_flags):
        if canc == 1 or np.isnan(dt) or np.isnan(tx) or tx <= 0:
            wheels_off.append(np.nan)
        else:
            h = int(dt) // 100
            m = int(dt) % 100
            total_m = (h * 60 + m + int(tx)) % 1440
            wo = (total_m // 60) * 100 + (total_m % 60)
            wheels_off.append(wo)
    
    # Construct DataFrame with standard BTS column schema
    df = pd.DataFrame({
        'FL_DATE': fl_date_strs,
        'OP_UNIQUE_CARRIER': carriers,
        'ORIGIN': origins,
        'DEST': dests,
        'CRS_DEP_TIME': crs_dep_times,
        'DEP_TIME': dep_times,
        'DEP_DELAY': dep_delays,
        'TAXI_OUT': simulated_taxi,
        'WHEELS_OFF': wheels_off,
        'DISTANCE': distances,
        'CANCELLED': cancelled_flags,
        'DIVERTED': diverted_flags
    })
    
    # Sort chronologically by date and scheduled departure
    df['sort_key'] = pd.to_datetime(df['FL_DATE']) + pd.to_timedelta(df['CRS_DEP_TIME'] // 100, unit='h') + pd.to_timedelta(df['CRS_DEP_TIME'] % 100, unit='m')
    df = df.sort_values('sort_key').reset_index(drop=True).drop(columns=['sort_key'])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created at: {output_path}")
    print(f"Total records: {len(df):,}")
    print(f"Cancelled flights: {df['CANCELLED'].sum():,} ({df['CANCELLED'].mean()*100:.2f}%)")
    print(f"Diverted flights: {df['DIVERTED'].sum():,} ({df['DIVERTED'].mean()*100:.2f}%)")
    print(f"Valid TAXI_OUT records: {df['TAXI_OUT'].notna().sum():,}")
    print(f"Extreme outliers (>120 min): {(df['TAXI_OUT'] > 120).sum():,}")
    print(f"DFW records: {(df['ORIGIN'] == 'DFW').sum():,} ({(df['ORIGIN'] == 'DFW').mean()*100:.2f}%)")
    return df

if __name__ == "__main__":
    generate_bts_data()
