import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
# 3. SELECT FEATURES
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

# Dataset is chronological.
# First 80% = training
# Last 20% = testing

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
# 6. DEFINE THREE ML MODELS
# =========================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=20,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )

}


# =========================================================
# 7. TRAIN AND EVALUATE MODELS
# =========================================================

results = []


for name, model in models.items():

    print("\n===================================")
    print("Training:", name)
    print("===================================")

    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )

        ]

    )


    # Train
    pipeline.fit(
        X_train,
        y_train
    )


    # Predict
    predictions = pipeline.predict(
        X_test
    )


    # Metrics
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


    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R²  :", round(r2, 4))


    results.append({

        "Model": name,

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2

    })


# =========================================================
# 8. DISPLAY FINAL COMPARISON
# =========================================================

results_df = pd.DataFrame(results)


print("\n\n===================================")
print("MODEL COMPARISON")
print("===================================")

print(
    results_df.to_string(
        index=False
    )
)


# =========================================================
# 9. SAVE RESULTS
# =========================================================

results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)


print(
    "\nResults saved to:"
    " model_comparison_results.csv"
)