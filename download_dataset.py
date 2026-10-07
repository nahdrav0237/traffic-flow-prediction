from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

print("Downloading traffic dataset...")

dataset = fetch_ucirepo(id=492)

X = dataset.data.features
y = dataset.data.targets

data = pd.concat([X, y], axis=1)

os.makedirs("dataset", exist_ok=True)

data.to_csv("dataset/traffic_data.csv", index=False)

print("\nDataset downloaded successfully!")
print("Rows:", len(data))
print("Columns:", len(data.columns))

print("\nColumns:")
print(data.columns.tolist())

print("\nFirst 5 rows:")
print(data.head())

print("\nMissing values:")
print(data.isnull().sum())