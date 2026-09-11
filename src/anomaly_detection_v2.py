import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


# ==========================================
# 1. LOAD FEATURE-ENGINEERED DATA
# ==========================================

df = pd.read_csv("data/weather_feature_engineered.csv")

df["Date/Time"] = pd.to_datetime(
    df["Date/Time"],
    format="mixed"
)

df = df.sort_values("Date/Time").reset_index(drop=True)

print("Dataset shape:", df.shape)


# ==========================================
# 2. SELECT FEATURES
# ==========================================

feature_columns = [
    "Temp_C",
    "Rel Hum_%",
    "Press_kPa",

    "Temp_Change",
    "Humidity_Change",
    "Pressure_Change",

    "Temp_Rolling_Mean",
    "Humidity_Rolling_Mean",
    "Pressure_Rolling_Mean",

    "Temp_Rolling_Std",
    "Humidity_Rolling_Std",
    "Pressure_Rolling_Std"
]


# ==========================================
# 3. PREPARE DATA
# ==========================================

model_data = df[feature_columns].dropna()

X = model_data


# ==========================================
# 4. TRAIN ISOLATION FOREST
# ==========================================

model = IsolationForest(
    n_estimators=200,
    contamination=0.01,
    random_state=42
)

model.fit(X)


# ==========================================
# 5. CALCULATE ANOMALY SCORE
# ==========================================

scores = model.decision_function(X)

df.loc[model_data.index, "Anomaly_Score"] = scores


# ==========================================
# 6. ANOMALY DECISION
# ==========================================

THRESHOLD = 0.0

df["ML_Anomaly"] = 0

df.loc[
    model_data.index,
    "ML_Anomaly"
] = (
    df.loc[model_data.index, "Anomaly_Score"] < THRESHOLD
).astype(int)


# ==========================================
# 7. CREATE OUTPUT
# ==========================================

output_columns = [
    "Date/Time",
    "Temp_C",
    "Rel Hum_%",
    "Press_kPa",

    "Temp_Change",
    "Humidity_Change",
    "Pressure_Change",

    "Temp_Rolling_Mean",
    "Humidity_Rolling_Mean",
    "Pressure_Rolling_Mean",

    "Temp_Rolling_Std",
    "Humidity_Rolling_Std",
    "Pressure_Rolling_Std",

    "Anomaly_Score",
    "ML_Anomaly"
]

results = df[output_columns].copy()


# ==========================================
# 8. SAVE OUTPUT
# ==========================================

results.to_csv(
    "anomaly_results_v2.csv",
    index=False
)


# ==========================================
# 9. SUMMARY
# ==========================================

print("\n==========================================")
print("SKYGUARD AI — V2 ANOMALY DETECTION")
print("==========================================")

print("Total observations:", len(results))

print(
    "Detected anomalies:",
    results["ML_Anomaly"].sum()
)

print(
    "Normal observations:",
    (results["ML_Anomaly"] == 0).sum()
)

print("\nOutput columns:")
print(results.columns.tolist())

print("\nSaved:")
print("anomaly_results_v2.csv")
