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

# STEP 4: 5-FOLD CV

# =====================================================

kf = KFold(
n_splits=5,
shuffle=True,
random_state=42
)

results = []

for model_name, model in models.items():

    print(f"\nTraining {model_name}")

    r2_scores = []
    rmse_scores = []
    mae_scores = []
    evs_scores = []

    for train_idx, test_idx in kf.split(X):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2_scores.append(
            r2_score(y_test, y_pred)
        )

        rmse_scores.append(
            np.sqrt(
                mean_squared_error(
                    y_test,
                    y_pred
                )
            )
        )

        mae_scores.append(
            mean_absolute_error(
                y_test,
                y_pred
            )
        )

        evs_scores.append(
            explained_variance_score(
                y_test,
                y_pred
            )
        )

    results.append([
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

# STEP 5: RESULTS TABLE

# =====================================================

results_df = pd.DataFrame(


results,

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

results_df = results_df.sort_values(
by="Mean_R2",
ascending=False
)

print(results_df)

# =====================================================

# STEP 6: SAVE

# =====================================================

results_df.to_csv(
"5Fold_Model_Comparison.csv",
index=False
)

print(
"\nResults saved as "
"5Fold_Model_Comparison.csv"
)
