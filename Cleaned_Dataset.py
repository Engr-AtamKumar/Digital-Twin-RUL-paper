import pandas as pd

# STEP 1: Load the CSV file
file_path = "datapoint_named_example.csv"  # Adjust path if needed
df = pd.read_csv(file_path)

# STEP 2: Convert 'timestamp' column to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# STEP 3: Sort data by timestamp (important for interpolation)
df = df.sort_values(by='timestamp')

# STEP 4: Interpolate missing numeric values
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')

# STEP 5: Drop rows with any remaining missing values
df_cleaned = df.dropna()

# STEP 6: (Optional) Save cleaned data to a new CSV file
df_cleaned.to_csv("cleaned_printer_data.csv", index=False)

# STEP 7: Preview cleaned data
print("Cleaned dataset preview:")
print(df_cleaned.head())
