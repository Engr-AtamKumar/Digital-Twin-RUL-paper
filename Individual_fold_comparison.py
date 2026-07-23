# =====================================================
# STEP 0: IMPORTS
# =====================================================

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)

from xgboost import XGBRegressor

from sqlalchemy import create_engine

from sklearn.model_selection import KFold

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score
)

import pandas as pd
import numpy as np

# =====================================================
# STEP 1: LOAD DATA
# =====================================================

print("Connecting to TimescaleDB...")

engine = create_engine(
    "postgresql+psycopg2://postgres:atam123@localhost:5432/printer"
)

df = pd.read_sql(
    "SELECT * FROM engineered_features",
    engine
)

print(f"Data loaded: {len(df)} rows")

# =====================================================
# STEP 2: FEATURES
# =====================================================

features = [
    'T_Hotend_Actual_Smooth',
    'Power_Hotend_Smooth',
    'hour',
    'dayofweek',
    'utilization'
]

target = 'RUL_simulated'

df = df.dropna(
    subset=features + [target]
)

X = df[features]
y = df[target]

# Log transform target

y = np.log1p(y)

# =====================================================
# STEP 3: MODELS
# =====================================================

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
            random_state=42,
            n_jobs=-1
        ),

    "Extra Trees":
        ExtraTreesRegressor(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
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

# =====================================================
# STEP 4: 5-FOLD CROSS VALIDATION
# =====================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

summary_results = []

fold_results = []

prediction_results = []

# =====================================================
# STEP 5: TRAINING
# =====================================================

for model_name, model in models.items():

    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    r2_scores = []
    rmse_scores = []
    mae_scores = []
    evs_scores = []

    for fold_no, (train_idx, test_idx) in enumerate(
        kf.split(X),
        start=1
    ):

        print(f"Fold {fold_no}")

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # -----------------------------------
        # Train Model
        # -----------------------------------

        model.fit(
            X_train,
            y_train
        )

        # -----------------------------------
        # Predict
        # -----------------------------------

        y_pred = model.predict(X_test)

        # -----------------------------------
        # Metrics
        # -----------------------------------

        r2 = r2_score(
            y_test,
            y_pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                y_pred
            )
        )

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        evs = explained_variance_score(
            y_test,
            y_pred
        )

        # -----------------------------------
        # Store Fold Metrics
        # -----------------------------------

        fold_results.append({

            "Model":
                model_name,

            "Fold":
                fold_no,

            "R2":
                r2,

            "RMSE":
                rmse,

            "MAE":
                mae,

            "EVS":
                evs
        })

        # -----------------------------------
        # Store Actual vs Predicted
        # -----------------------------------

        temp_predictions = pd.DataFrame({

            "Model":
                model_name,

            "Fold":
                fold_no,

            "Actual":
                y_test.values,

            "Predicted":
                y_pred

        })

        prediction_results.append(
            temp_predictions
        )

        # -----------------------------------
        # Store for Mean & Std
        # -----------------------------------

        r2_scores.append(r2)

        rmse_scores.append(rmse)

        mae_scores.append(mae)

        evs_scores.append(evs)

    # =====================================
    # Mean & Std Across Folds
    # =====================================

    summary_results.append([

        model_name,

        np.mean(r2_scores),
        np.std(r2_scores),

        np.mean(rmse_scores),
        np.std(rmse_scores),

        np.mean(mae_scores),
        np.std(mae_scores),

        np.mean(evs_scores),
        np.std(evs_scores)

    ])

# =====================================================
# STEP 6: SUMMARY TABLE
# =====================================================

summary_df = pd.DataFrame(

    summary_results,

    columns=[

        "Model",

        "Mean_R2",
        "Std_R2",

        "Mean_RMSE",
        "Std_RMSE",

        "Mean_MAE",
        "Std_MAE",

        "Mean_EVS",
        "Std_EVS"
    ]
)

summary_df = summary_df.sort_values(
    by="Mean_R2",
    ascending=False
)

print("\n")
print("="*80)
print("SUMMARY RESULTS")
print("="*80)

print(summary_df)

# =====================================================
# STEP 7: SAVE SUMMARY
# =====================================================

summary_df.to_csv(
    "5Fold_Model_Comparison_updated.csv",
    index=False
)

# =====================================================
# STEP 8: SAVE FOLD-WISE METRICS
# =====================================================

fold_results_df = pd.DataFrame(
    fold_results
)

fold_results_df.to_csv(
    "Fold_Wise_Metrics.csv",
    index=False
)

# =====================================================
# STEP 9: SAVE ALL PREDICTIONS
# =====================================================

prediction_df = pd.concat(
    prediction_results,
    ignore_index=True
)

prediction_df.to_csv(
    "All_Folds_Actual_vs_Predicted.csv",
    index=False
)

# =====================================================
# STEP 10: FINISHED
# =====================================================

print("\nFiles Created Successfully")

print("1. 5Fold_Model_Comparison.csv")
print("2. Fold_Wise_Metrics.csv")
print("3. All_Folds_Actual_vs_Predicted.csv")