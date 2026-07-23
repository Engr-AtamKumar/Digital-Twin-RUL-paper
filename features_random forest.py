# =====================================================
# STEP 0: IMPORTS
# =====================================================

import pandas as pd
import numpy as np

from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt

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

print(f"Data Loaded: {len(df)} rows")

# =====================================================
# STEP 2: FEATURES AND TARGET
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

# =====================================================
# STEP 3: TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =====================================================
# STEP 4: LOG TRANSFORMATION
# =====================================================

y_train = np.log1p(y_train)

# =====================================================
# STEP 5: RANDOM FOREST MODEL
# =====================================================

rf = RandomForestRegressor(
    n_estimators=50,
    n_jobs=-1,
    random_state=42
)

print("Training Random Forest...")

rf.fit(X_train, y_train)

print("Training Complete")

# =====================================================
# STEP 6: FEATURE IMPORTANCE
# =====================================================

importance_df = pd.DataFrame({

    "Feature": features,

    "Importance": rf.feature_importances_

})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)

print("\nFeature Importance")

print(importance_df)

# =====================================================
# STEP 7: SAVE FEATURE IMPORTANCE
# =====================================================

importance_df.to_csv(
    "RandomForest_Feature_Importance.csv",
    index=False
)

print(
    "\nFeature importance saved as "
    "RandomForest_Feature_Importance.csv"
)

# =====================================================
# STEP 8: FEATURE IMPORTANCE PLOT
# =====================================================

plt.figure(figsize=(9,6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance Score")

plt.ylabel("Features")

plt.title(
    "Feature Importance of Random Forest Model"
)

plt.tight_layout()

plt.savefig(
    "Figure_Feature_Importance_RF.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nFigure saved as "
    "Figure_Feature_Importance_RF.png"
)