import pandas as pd
from sqlalchemy import create_engine

# Create SQLAlchemy engine
engine = create_engine("postgresql+psycopg2://postgres:atam123@localhost:5432/printer")

# SQL query
query = "SELECT * FROM printer_data ORDER BY timestamp;"

# Run query using SQLAlchemy connection
df = pd.read_sql(query, engine)

# Show result
print(df.head())
print(f"Total rows loaded: {len(df)}")
print(df.head())  # Optional: still shows just the first 5 rows
df.to_csv("printer_data_export.csv", index=False)


