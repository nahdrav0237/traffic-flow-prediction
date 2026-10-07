import pandas as pd
import numpy as np
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

print("Loading dataset...")

data = pd.read_csv("dataset/traffic_data.csv")

print("Rows:", len(data))


# --------------------------------------------------
# 2. DATE/TIME PROCESSING
# --------------------------------------------------

data["date_time"] = pd.to_datetime(data["date_time"])

# Extract useful time features
data["hour"] = data["date_time"].dt.hour
data["day_of_week"] = data["date_time"].dt.dayofweek
data["month"] = data["date_time"].dt.month
data["year"] = data["date_time"].dt.year


# --------------------------------------------------
# 3. HANDLE HOLIDAY
# --------------------------------------------------

data["holiday"] = data["holiday"].fillna("None")


# --------------------------------------------------
# 4. SELECT FEATURES
# --------------------------------------------------

features = [
    "hour",
    "day_of_week",
    "month",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "holiday",
    "weather_main",
    "weather_description"
]

target = "traffic_volume"


X = data[features]
y = data[target]


# --------------------------------------------------
# 5. TIME-BASED TRAIN/TEST SPLIT
# --------------------------------------------------

# Dataset is already chronological.
# First 80% = training
# Last 20% = testing

split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# --------------------------------------------------
# 6. CATEGORICAL + NUMERICAL FEATURES
# --------------------------------------------------

categorical_features = [
    "holiday",
    "weather_main",
    "weather_description"
]

numerical_features = [
    "hour",
    "day_of_week",
    "month",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all"
]


# --------------------------------------------------
# 7. PREPROCESSING
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# --------------------------------------------------
# 8. ML MODEL
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=20,
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 9. CREATE PIPELINE
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# --------------------------------------------------
# 10. TRAIN
# --------------------------------------------------

print("\nTraining Random Forest model...")

pipeline.fit(X_train, y_train)

print("Training completed!")


# --------------------------------------------------
# 11. PREDICTION
# --------------------------------------------------

print("\nTesting model...")

predictions = pipeline.predict(X_test)


# --------------------------------------------------
# 12. EVALUATION
# --------------------------------------------------

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 4))


# --------------------------------------------------
# 13. SAVE MODEL
# --------------------------------------------------

os.makedirs("model", exist_ok=True)

joblib.dump(
    pipeline,
    "model/traffic_flow_model.pkl"
)

print("\nModel saved successfully!")
print("File: model/traffic_flow_model.pkl")