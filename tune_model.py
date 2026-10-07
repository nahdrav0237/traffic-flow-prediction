import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib


# =========================================================
# 1. LOAD DATASET
# =========================================================

print("Loading dataset...")

df = pd.read_csv("dataset/traffic_data.csv")

print("Rows:", len(df))


# =========================================================
# 2. FEATURE ENGINEERING
# =========================================================

df["date_time"] = pd.to_datetime(df["date_time"])

df["hour"] = df["date_time"].dt.hour
df["day_of_week"] = df["date_time"].dt.dayofweek
df["month"] = df["date_time"].dt.month


# =========================================================
# 3. FEATURES AND TARGET
# =========================================================

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

X = df[features]
y = df[target]


# =========================================================
# 4. TRAIN / TEST SPLIT
# =========================================================

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# =========================================================
# 5. PREPROCESSING
# =========================================================

numeric_features = [
    "hour",
    "day_of_week",
    "month",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all"
]

categorical_features = [
    "holiday",
    "weather_main",
    "weather_description"
]


preprocessor = ColumnTransformer(

    transformers=[

        (
            "num",
            "passthrough",
            numeric_features
        ),

        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )

    ]

)


# =========================================================
# 6. RANDOM FOREST PIPELINE
# =========================================================

pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            )
        )

    ]

)


# =========================================================
# 7. HYPERPARAMETER GRID
# =========================================================

param_grid = {

    "model__n_estimators": [
        20,
        50,
        100
    ],

    "model__max_depth": [
        None,
        10,
        20
    ],

    "model__min_samples_split": [
        2,
        5
    ],

    "model__min_samples_leaf": [
        1,
        2
    ]

}


# =========================================================
# 8. GRID SEARCH
# =========================================================

print("\nStarting hyperparameter tuning...")

print("This may take several minutes.")

grid_search = GridSearchCV(

    estimator=pipeline,

    param_grid=param_grid,

    scoring="neg_mean_absolute_error",

    cv=3,

    n_jobs=-1,

    verbose=1

)


grid_search.fit(
    X_train,
    y_train
)


# =========================================================
# 9. BEST PARAMETERS
# =========================================================

print("\n===================================")
print("BEST PARAMETERS")
print("===================================")

print(
    grid_search.best_params_
)


print("\nBest CV MAE:")

print(
    round(
        -grid_search.best_score_,
        2
    )
)


# =========================================================
# 10. TEST BEST MODEL
# =========================================================

best_model = grid_search.best_estimator_


predictions = best_model.predict(
    X_test
)


# =========================================================
# 11. FINAL PERFORMANCE
# =========================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


print("\n===================================")
print("OPTIMIZED MODEL PERFORMANCE")
print("===================================")

print(
    "MAE :",
    round(mae, 2)
)

print(
    "RMSE:",
    round(rmse, 2)
)

print(
    "R²  :",
    round(r2, 4)
)

print(
    "R² %:",
    round(r2 * 100, 2),
    "%"
)


# =========================================================
# 12. SAVE OPTIMIZED MODEL
# =========================================================

joblib.dump(
    best_model,
    "model/traffic_flow_model_optimized.pkl"
)


print("\n===================================")
print("MODEL SAVED")
print("===================================")

print(
    "File: model/traffic_flow_model_optimized.pkl"
)