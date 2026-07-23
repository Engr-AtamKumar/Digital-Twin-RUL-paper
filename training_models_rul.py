# ============================================================
# STEP 0: IMPORTS
# ============================================================

import pandas as pd
import numpy as np

from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)

from xgboost import XGBRegressor

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    explained_variance_score
)

# ============================================================
# STEP 1: LOAD DATA FROM TIMESCALEDB
# ============================================================

print("🔄 Connecting to TimescaleDB...")

engine = create_engine(
    "postgresql+psycopg2://postgres:atam123@localhost:5432/printer"
)

df = pd.read_sql(
    "SELECT * FROM engineered_features",
    engine
)

print(f"✅ Data loaded: {len(df)} rows")

# ============================================================
# STEP 2: DEFINE FEATURES AND TARGET
# ============================================================

features = [
    'T_Hotend_Actual_Smooth',
    'Power_Hotend_Smooth',
    'hour',
    'dayofweek',
    'utilization'
]

target = 'RUL_simulated'

# Remove missing values

df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# ============================================================
# STEP 3: TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train_original, y_test_original = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(
    f"📊 Training samples: {len(X_train)}, "
    f"Testing samples: {len(X_test)}"
)

# ============================================================
# STEP 4: LOG TRANSFORMATION OF TARGET
# ============================================================

y_train = np.log1p(y_train_original)
y_test = np.log1p(y_test_original)

# ============================================================
# STEP 5: DEFINE MODELS
# ============================================================

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

# ============================================================
# STEP 6: TRAIN MODELS & EVALUATE
# ============================================================

results = []

for name, model in models.items():

    try:

        print("\n" + "=" * 60)
        print(f"🚀 Training {name}")
        print("=" * 60)

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(X_train, y_train)

        # ----------------------------------------------------
        # Predict (LOG SCALE)
        # ----------------------------------------------------

        y_pred = model.predict(X_test)

        # ----------------------------------------------------
        # Metrics (LOG SCALE)
        # ----------------------------------------------------

        r2 = r2_score(y_test, y_pred)

        rmse = np.sqrt(
            mean_squared_error(y_test, y_pred)
        )

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        mape = (
            mean_absolute_percentage_error(
                y_test,
                y_pred
            ) * 100
        )

        mbe = np.mean(
            y_pred - y_test
        )

        evs = explained_variance_score(
            y_test,
            y_pred
        )

        # ----------------------------------------------------
        # Save Results
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # SAVE ACTUAL vs PREDICTED FOR RANDOM FOREST
        # ----------------------------------------------------

        if name == "Random Forest":

            actual_rul = np.expm1(y_test)

            predicted_rul = np.expm1(y_pred)

            rf_predictions = pd.DataFrame({

                "Actual_RUL":
                    actual_rul,

                "Predicted_RUL":
                    predicted_rul

            })

            rf_predictions.to_csv(
                "RandomForest_Actual_vs_Predicted_RUL.csv",
                index=False
            )

            print(
                "✅ Random Forest Actual vs Predicted "
                "RUL file saved"
            )

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

# ============================================================
# STEP 7: CREATE MODEL COMPARISON TABLE
# ============================================================

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

# ============================================================
# STEP 8: DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(results_df)

# ============================================================
# STEP 9: SAVE MODEL COMPARISON
# ============================================================

results_df.to_csv(
    "model_comparison_results_updated.csv",
    index=False
)

print(
    "\n📁 Results saved as "
    "updated_model_comparison_results.csv"
)

print(
    "📁 Random Forest predictions saved as "
    "RandomForest_Actual_vs_Predicted_RUL.csv"
)