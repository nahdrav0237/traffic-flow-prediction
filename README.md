# 🚦 Traffic Flow Prediction

A machine-learning-based web application that predicts traffic volume using time, weather, and holiday information. The project combines an optimized Random Forest Regression model with a Flask backend and an interactive Leaflet/OpenStreetMap interface.

## 🌐 Live Demo

**Application:** https://traffic-flow-prediction-vt0y.onrender.com

**GitHub Repository:** https://github.com/nahdrav0237/traffic-flow-prediction

> The Render free service may sleep after periods of inactivity, so the first request can take a little longer.

## 📌 Project Overview

Traffic congestion is influenced by several factors such as time of day, day of the week, weather conditions, cloud cover, rainfall, snowfall, and holidays.

This project uses historical traffic data to learn these relationships and predict the expected traffic volume in **vehicles per hour**.

The application provides:

- Machine learning-based traffic prediction
- Interactive map using Leaflet and OpenStreetMap
- Location search
- Current-location selection through browser geolocation
- Clickable map locations
- Prediction summary with selected coordinates
- Traffic condition classification
- Saved and reloadable optimized ML model

### Important note about location

Latitude and longitude are used by the website for **map visualization and location context**. They are **not input features to the ML model**, because the UCI dataset used for training does not provide arbitrary GPS coordinates for prediction locations.

The actual ML prediction is based on the time, weather, and holiday features described below.

## 🎯 Objectives

1. Obtain a real-world traffic dataset.
2. Perform data preprocessing and feature engineering.
3. Train multiple machine learning models.
4. Compare model performance.
5. Optimize the best-performing model using hyperparameter tuning.
6. Save the optimized model using Joblib.
7. Build a Flask web application around the trained model.
8. Provide an interactive map and location-selection interface.
9. Make predictions on unseen input data without retraining.

## 📊 Dataset

### UCI Metro Interstate Traffic Volume

Source: **UCI Machine Learning Repository**

Dataset: **Metro Interstate Traffic Volume**

Official source: https://archive.ics.uci.edu/dataset/492/metro%2Binterstate%2Btraffic%2Bvolume

The dataset contains **48,204 observations**.

Original columns include:

- `holiday`
- `temp`
- `rain_1h`
- `snow_1h`
- `clouds_all`
- `weather_main`
- `weather_description`
- `date_time`
- `traffic_volume`

## 🧠 Machine Learning Pipeline

```text
Real-World Dataset
        ↓
Data Cleaning
        ↓
Date/Time Feature Extraction
        ↓
Feature Engineering
        ↓
Train/Test Split
        ↓
Multiple ML Models
        ↓
Model Comparison
        ↓
Hyperparameter Tuning
        ↓
Optimized Random Forest
        ↓
Save Model with Joblib
        ↓
Flask Prediction API
        ↓
Web Interface
        ↓
Traffic Prediction
```

## 🔧 Features Used by the Final Model

### Numerical Features

- `hour`
- `day_of_week`
- `month`
- `temp`
- `rain_1h`
- `snow_1h`
- `clouds_all`

### Categorical Features

- `holiday`
- `weather_main`
- `weather_description`

The target variable is:

```text
traffic_volume
```

which represents traffic volume in vehicles per hour.

## 🤖 Models Compared

Three regression models were evaluated:

### 1. Linear Regression

Used as a baseline model.

**R² Score: 16.79%**

### 2. Random Forest Regression

Performed substantially better than Linear Regression.

**R² Score: 91.98%**

### 3. Gradient Boosting Regression

Also performed well.

**R² Score: 91.16%**

Based on the initial comparison, Random Forest Regression was selected for further optimization.

## ⚙️ Hyperparameter Tuning

GridSearchCV was used to optimize the Random Forest model.

The selected parameters were:

```text
n_estimators = 100
max_depth = 10
min_samples_split = 2
min_samples_leaf = 2
```

The optimized model achieved:

| Metric | Result |
|---|---:|
| R² Score | **92.84%** |
| MAE | **290.86 vehicles/hour** |
| RMSE | **526.45 vehicles/hour** |

### Interpretation

- **R² = 92.84%** indicates that the model explains a large proportion of the variation in traffic volume in the test data.
- **MAE = 290.86 vehicles/hour** means the average absolute prediction error is about 291 vehicles per hour.
- **RMSE = 526.45 vehicles/hour** gives greater weight to larger prediction errors.

## 🗺️ Web Application

The frontend is built using:

- HTML
- CSS
- JavaScript
- Leaflet
- OpenStreetMap

The application provides:

### Location Search

Users can search for a city or location. The application uses OpenStreetMap's Nominatim search service to locate the place on the map.

### Map Selection

Users can click anywhere on the map to select a location.

### Current Location

The browser's geolocation API can be used to select the user's current location.

### Prediction Summary

The application displays:

- Selected location
- Latitude
- Longitude
- Prediction time
- Predicted traffic volume
- Traffic condition

### Traffic Conditions

Predicted traffic is categorized as:

```text
Low
Moderate
High
Very High
```

## 🏗️ Project Structure

```text
traffic-flow-prediction/
│
├── app.py
├── requirements.txt
├── Procfile
│
├── compare_models.py
├── tune_model.py
├── create_performance_visulizations.py
│
├── model_comparison_results.csv
├── optimized_model_metrics.csv
│
├── r2_comparison.png
├── mae_comparison.png
├── rmse_comparison.png
├── actual_vs_predicted.png
│
├── dataset/
│   └── traffic_data.csv
│
├── model/
│   ├── traffic_flow_model.pkl
│   └── traffic_flow_model_optimized.pkl
│
└── templates/
    └── index.html
```

## 💻 Technologies Used

### Programming

- Python
- JavaScript
- HTML
- CSS

### Machine Learning

- Pandas
- NumPy
- Scikit-learn
- Joblib

### Backend

- Flask
- Flask-CORS
- Gunicorn

### Frontend / Maps

- Leaflet
- OpenStreetMap
- Nominatim

### Deployment

- GitHub
- Render

## 🚀 Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/nahdrav0237/traffic-flow-prediction.git
cd traffic-flow-prediction
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Start the Flask application

```powershell
python app.py
```

The application will be available at:

```text
http://127.0.0.1:5000
```

## 🔮 Making a Prediction

The web application sends the selected input features to the Flask `/predict` endpoint.

Example input:

```json
{
    "hour": 17,
    "day_of_week": 2,
    "month": 10,
    "temp": 290,
    "rain_1h": 0,
    "snow_1h": 0,
    "clouds_all": 20,
    "holiday": "None",
    "weather_main": "Clear",
    "weather_description": "sky is clear"
}
```

The backend loads the saved optimized model and returns the predicted traffic volume and traffic condition.

The model is **not retrained when a user makes a prediction**.

## 💾 Saved Model

The optimized model is stored at:

```text
model/traffic_flow_model_optimized.pkl
```

It is loaded by the Flask application using Joblib.

This allows the deployed application to make predictions directly from the saved trained pipeline.

## 📈 Performance Visualizations

The project includes:

- `r2_comparison.png`
- `mae_comparison.png`
- `rmse_comparison.png`
- `actual_vs_predicted.png`

These visualizations are used to compare the models and analyze the optimized model's performance.

## 🌍 Real-World Application

Traffic flow prediction can support:

- Traffic management
- Congestion monitoring
- Route planning
- Intelligent transportation systems
- Urban planning
- Road capacity analysis
- Travel-time planning

A production system could be extended by connecting real-time weather and traffic sources and retraining the model with additional geographic and road-level information.

## ⚠️ Current Limitations

This project is a machine learning prediction prototype based on the UCI Metro Interstate Traffic Volume dataset.

Current limitations include:

- The training dataset is historical rather than live traffic data.
- The map does not provide live traffic congestion information.
- Latitude and longitude are used for visualization, not as ML prediction features.
- Weather values are currently entered through the application rather than automatically retrieved from a live weather API.
- Predictions represent learned relationships from the training dataset and may not generalize equally to every geographic location.

## 🔮 Future Enhancements

Possible improvements include:

1. Automatic live weather retrieval.
2. Real-time traffic data integration.
3. Road-specific traffic prediction.
4. Geographic features such as road type and location.
5. Traffic prediction for multiple future time periods.
6. Interactive traffic heatmaps.
7. Historical prediction charts.
8. User-selected date and time forecasting.
9. Model retraining with newer traffic data.
10. More advanced models such as XGBoost or neural networks.

## 👨‍💻 Academic Project

This project demonstrates the complete machine learning lifecycle:

```text
Problem Definition
       ↓
Data Collection
       ↓
EDA
       ↓
Preprocessing
       ↓
Feature Engineering
       ↓
Model Training
       ↓
Model Comparison
       ↓
Hyperparameter Tuning
       ↓
Model Evaluation
       ↓
Model Saving
       ↓
Flask Integration
       ↓
Web Deployment
```

It demonstrates building, evaluating, optimizing, saving, and deploying a machine learning model for a real-world traffic prediction problem.

## 📜 License

This project is intended for educational and academic purposes.
