import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# 1. LOAD MODEL COMPARISON RESULTS
# =========================================================

results = pd.read_csv(
    "model_comparison_results.csv"
)


# =========================================================
# 2. R² COMPARISON
# =========================================================

plt.figure(figsize=(9, 5))

plt.bar(
    results["Model"],
    results["R2"] * 100
)

plt.ylabel("R² Score (%)")

plt.title(
    "Model Performance Comparison - R²"
)

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "r2_comparison.png",
    dpi=200
)

plt.close()


# =========================================================
# 3. MAE COMPARISON
# =========================================================

plt.figure(figsize=(9, 5))

plt.bar(
    results["Model"],
    results["MAE"]
)

plt.ylabel(
    "MAE (vehicles/hour)"
)

plt.title(
    "Model Performance Comparison - MAE"
)

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "mae_comparison.png",
    dpi=200
)

plt.close()


# =========================================================
# 4. RMSE COMPARISON
# =========================================================

plt.figure(figsize=(9, 5))

plt.bar(
    results["Model"],
    results["RMSE"]
)

plt.ylabel(
    "RMSE (vehicles/hour)"
)

plt.title(
    "Model Performance Comparison - RMSE"
)

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "rmse_comparison.png",
    dpi=200
)

plt.close()


# =========================================================
# 5. LOAD OPTIMIZED MODEL
# =========================================================

print("Loading optimized model...")

optimized_model = joblib.load(
    "model/traffic_flow_model_optimized.pkl"
)


# =========================================================
# 6. LOAD DATASET
# =========================================================

df = pd.read_csv(
    "dataset/traffic_data.csv"
)


# =========================================================
# 7. FEATURE ENGINEERING
# =========================================================

df["date_time"] = pd.to_datetime(
    df["date_time"]
)

df["hour"] = df["date_time"].dt.hour

df["day_of_week"] = (
    df["date_time"].dt.dayofweek
)

df["month"] = (
    df["date_time"].dt.month
)


# =========================================================
# 8. FEATURES
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

X = df[features]

y = df["traffic_volume"]


# =========================================================
# 9. TEST DATA
# =========================================================

split_index = int(
    len(df) * 0.8
)

X_test = X.iloc[
    split_index:
]

y_test = y.iloc[
    split_index:
]


# =========================================================
# 10. PREDICTIONS
# =========================================================

print("Generating predictions...")

predictions = optimized_model.predict(
    X_test
)


# =========================================================
# 11. ACTUAL VS PREDICTED
# =========================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.25
)

min_value = min(
    y_test.min(),
    predictions.min()
)

max_value = max(
    y_test.max(),
    predictions.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value]
)

plt.xlabel(
    "Actual Traffic Volume"
)

plt.ylabel(
    "Predicted Traffic Volume"
)

plt.title(
    "Optimized Random Forest - Actual vs Predicted"
)

plt.tight_layout()

plt.savefig(
    "actual_vs_predicted.png",
    dpi=200
)

plt.close()


# =========================================================
# 12. FINAL METRICS
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


metrics = pd.DataFrame([{

    "Model":
        "Optimized Random Forest",

    "MAE":
        mae,

    "RMSE":
        rmse,

    "R2":
        r2,

    "R2_Percent":
        r2 * 100

}])


metrics.to_csv(
    "optimized_model_metrics.csv",
    index=False
)


# =========================================================
# 13. DISPLAY RESULTS
# =========================================================

print("\n===================================")
print("OPTIMIZED MODEL METRICS")
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


print("\n===================================")
print("VISUALIZATIONS CREATED")
print("===================================")

print("r2_comparison.png")
print("mae_comparison.png")
print("rmse_comparison.png")
print("actual_vs_predicted.png")

print("\nMetrics saved:")
print("optimized_model_metrics.csv")