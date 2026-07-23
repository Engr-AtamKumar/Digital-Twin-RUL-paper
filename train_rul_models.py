# === Step 0: Imports ===
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sqlalchemy import create_engine
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import pandas as pd
from sklearn.model_selection import train_test_split


from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    explained_variance_score
)

import numpy as np

# === Step 1: Load data from TimescaleDB ===
print("🔄 Connecting to TimescaleDB...")
engine = create_engine("postgresql+psycopg2://postgres:atam123@localhost:5432/printer")
df = pd.read_sql("SELECT * FROM engineered_features", engine)
print(f"✅ Data loaded: {len(df)} rows")

# === Step 2: Define features and target ===
features = ['T_Hotend_Actual_Smooth', 'Power_Hotend_Smooth', 'hour', 'dayofweek', 'utilization']
target = 'RUL_simulated'

# Remove rows with missing values in features or target
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# === Step 3: Train/Test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"📊 Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

y_train = np.log1p(y_train)
y_test = np.log1p(y_test)

# === Step 4: Train XGBoost Regressor ===
models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            max_depth=10,
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=50,
            n_jobs=-1,
            random_state=42
        ),

    "Extra Trees":
        ExtraTreesRegressor(
            n_estimators=50,
            n_jobs=-1,
            random_state=42
        ),

    "Hist Gradient Boosting":
        HistGradientBoostingRegressor(
            random_state=42
        ),

    "XGBoost":
        XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            tree_method="hist",
            random_state=42
        )
}
# === Step 5: Predictions & Evaluation ===
results = []

for name, model in models.items():

    try:

        print(f"\n{'='*60}")
        print(f"🚀 Training {name}")
        print(f"{'='*60}")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        mae = mean_absolute_error(y_test, y_pred)

        mape = mean_absolute_percentage_error(y_test, y_pred) * 100

        mbe = np.mean(y_pred - y_test)

        evs = explained_variance_score(y_test, y_pred)

        results.append([
            name,
            r2,
            rmse,
            mae,
            mape,
            mbe,
            evs
        ])

        print(f"✅ {name} completed successfully")
        print(f"R² = {r2:.4f}")
        print(f"RMSE = {rmse:.4f}")

    except Exception as e:

        print(f"\n❌ {name} FAILED")
        print(f"Reason: {str(e)}")

        results.append([
            name,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan
        ])
        
# === Step 6: Create Results Table ===

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "R2",
        "RMSE",
        "MAE",
        "MAPE (%)",
        "Bias Error (MBE)",
        "Explained Variance"
    ]
)

results_df = results_df.sort_values(
    by="R2",
    ascending=False,
    na_position="last"
)

print("\n")
print("="*80)
print("MODEL COMPARISON")
print("="*80)
print(results_df)

# === Step 7: Save Results ===

results_df.to_csv(
    "model_comparison_results_updated.csv",
    index=False
)

print("\n📁 Results saved as model_comparison_results.csv")