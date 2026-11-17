
from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load trained ML model and feature list
MODEL_PATH = "models/car_price_model.pkl"
FEATURE_PATH = "models/features.pkl"

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURE_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Collect car details
        car_data = {
            "name": request.form["name"],
            "year": int(request.form["year"]),
            "km_driven": int(request.form["km_driven"]),
            "fuel": request.form["fuel"],
            "seller_type": request.form["seller_type"],
            "transmission": request.form["transmission"],
            "owner": request.form["owner"],
            "mileage": float(request.form["mileage"]),
            "engine": float(request.form["engine"]),
            "max_power": float(request.form["max_power"]),
            "seats": float(request.form["seats"])
        }

        # Convert input into DataFrame
        input_data = pd.DataFrame([car_data])

        # Maintain training feature order
        input_data = input_data[features]

        # Generate actual prediction
        predicted_price = model.predict(input_data)[0]

        # Prepare result
        return render_template(
            "index.html",
            prediction=round(max(0, predicted_price)),
            car_name=car_data["name"],
            car_year=car_data["year"],
            km_driven=car_data["km_driven"],
            fuel=car_data["fuel"],
            transmission=car_data["transmission"]
        )

    except Exception as error:
        return render_template(
            "index.html",
            error=str(error)
        ), 400


if __name__ == "__main__":
    app.run(debug=True)
