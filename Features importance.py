import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# =====================================================
# LOAD DATA
# =====================================================

print("Loading data...")

engine = create_engine(
    "postgresql+psycopg2://postgres:atam123@localhost:5432/printer"
)

df = pd.read_sql(
    "SELECT * FROM engineered_features",
    engine
)

# =====================================================
# FEATURES
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

y = np.log1p(
    df[target]
)

# =====================================================
# RANDOM FOREST MODEL
# =====================================================

print("Training Random Forest...")

rf = RandomForestRegressor(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

rf.fit(X, y)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance_df = pd.DataFrame({

    "Feature": features,
    "Importance": rf.feature_importances_

})

# =====================================================
# RENAME FEATURES FOR PUBLICATION
# =====================================================

feature_labels = {

    "T_Hotend_Actual_Smooth":
        "Hotend Temperature",

    "Power_Hotend_Smooth":
        "Power Consumption",

    "hour":
        "Operating Hour",

    "dayofweek":
        "Operating Day",

    "utilization":
        "Machine Utilization"
}

importance_df["Feature"] = (
    importance_df["Feature"]
    .map(feature_labels)
)

# =====================================================
# CONVERT TO PERCENTAGE
# =====================================================

importance_df["Importance (%)"] = (
    importance_df["Importance"] * 100
)

# =====================================================
# SORT ASCENDING FOR LOLLIPOP
# =====================================================

importance_df = importance_df.sort_values(
    by="Importance (%)",
    ascending=True
)

print("\nFeature Importance (%)")
print(
    importance_df[
        ["Feature", "Importance (%)"]
    ]
)

# =====================================================
# LOLLIPOP CHART
# =====================================================

plt.figure(
    figsize=(10, 6)
)

# Stem Lines

plt.hlines(

    y=importance_df["Feature"],

    xmin=0,

    xmax=importance_df["Importance (%)"],

    color="lightgray",

    linewidth=3

)

# Marker Colors

colors = [

    "#E63946",   # Red
    "#F4A261",   # Orange
    "#2A9D8F",   # Green
    "#457B9D",   # Blue
    "#1D3557"    # Dark Blue

]

# Lollipop Heads

plt.scatter(

    importance_df["Importance (%)"],

    importance_df["Feature"],

    s=450,

    c=colors,

    edgecolors="black",

    linewidths=1.5,

    zorder=3

)

# Percentage Labels

for _, row in importance_df.iterrows():

    plt.text(

        row["Importance (%)"] + 0.8,

        row["Feature"],

        f'{row["Importance (%)"]:.2f}%',

        va='center',

        fontsize=11,

        fontweight='bold'

    )

# =====================================================
# FORMATTING
# =====================================================

plt.xlabel(
    "Feature Importance (%)",
    fontsize=12,
    fontweight="bold"
)

plt.ylabel("")

plt.title(
    "Random Forest Feature Importance Analysis",
    fontsize=15,
    fontweight="bold"
)

plt.grid(
    axis='x',
    linestyle='--',
    alpha=0.4
)

plt.xlim(
    0,
    importance_df["Importance (%)"].max() + 8
)

plt.tight_layout()

# =====================================================
# SAVE FIGURE
# =====================================================

plt.savefig(
    "Figure7_Lollipop_Feature_Importance.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

print(
    "\nFigure saved as:"
)
print(
    "Figure7_Lollipop_Feature_Importance.png"
)