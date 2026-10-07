from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)

CORS(app)


# ==========================================
# LOAD ML MODEL
# ==========================================

model = joblib.load(
    "model/traffic_flow_model.pkl"
)


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================
# PREDICTION API
# ==========================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():


    data = request.json


    # Create model input

    input_data = pd.DataFrame([{

        "hour":
            int(data["hour"]),

        "day_of_week":
            int(data["day_of_week"]),

        "month":
            int(data["month"]),

        "temp":
            float(data["temp"]),

        "rain_1h":
            float(data["rain_1h"]),

        "snow_1h":
            float(data["snow_1h"]),

        "clouds_all":
            float(data["clouds_all"]),

        "holiday":
            data["holiday"],

        "weather_main":
            data["weather_main"],

        "weather_description":
            data["weather_description"]

    }])


    # ======================================
    # ML PREDICTION
    # ======================================

    prediction = model.predict(
        input_data
    )[0]


    prediction = max(
        0,
        round(float(prediction))
    )


    # ======================================
    # TRAFFIC LEVEL
    # ======================================

    if prediction < 2500:

        traffic_level = "Low"

    elif prediction < 4500:

        traffic_level = "Moderate"

    elif prediction < 6000:

        traffic_level = "High"

    else:

        traffic_level = "Very High"



    # ======================================
    # RETURN RESULT
    # ======================================

    return jsonify({

        "predicted_traffic":
            prediction,

        "traffic_level":
            traffic_level

    })



# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )