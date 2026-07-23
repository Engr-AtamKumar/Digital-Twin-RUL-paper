# === Step 0: Imports ===
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sqlalchemy import create_engine

# === Step 1: Load data from TimescaleDB ===
engine = create_engine("postgresql+psycopg2://postgres:atam123@localhost:5432/printer")
query = "SELECT * FROM printer_data ORDER BY timestamp;"
df = pd.read_sql(query, engine)

print(f"✅ Data loaded successfully: {len(df)} rows")

# === Step 2: Convert timestamp to datetime & extract time features ===
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['minute'] = df['timestamp'].dt.minute

# === Step 3: Handle missing values ===
df['T_Hotend_Actual'] = df['T_Hotend_Actual'].ffill()
df['Power_Hotend'] = df['Power_Hotend'].ffill()
df['E_Setpoint'] = df['E_Setpoint'].fillna(0)

# Drop rows if still missing
df.dropna(subset=['T_Hotend_Actual', 'Power_Hotend'], inplace=True)

# === Step 4: Smoothed rolling features ===
df['T_Hotend_Actual_Smooth'] = df['T_Hotend_Actual'].rolling(window=5).mean()
df['Power_Hotend_Smooth'] = df['Power_Hotend'].rolling(window=5).mean()

# === Step 5: Machine utilization ===
df['is_active'] = df['E_Setpoint'] > 0
df['utilization'] = df['is_active'].rolling(window=60).mean()

# === Step 6: Anomaly detection using z-score ===
df['hotend_zscore'] = zscore(df['T_Hotend_Actual'].ffill())
df['power_zscore'] = zscore(df['Power_Hotend'].ffill())

# === Step 7: Simulate Remaining Useful Life (RUL) ===
df = df.sort_values('timestamp')
df['RUL_simulated'] = len(df) - df.index.values

# === Step 8: Save to CSV ===
output_path = "features_engineered.csv"
df.to_csv(output_path, index=False)

print(f"✅ Feature engineering complete. Saved to: {output_path}")
