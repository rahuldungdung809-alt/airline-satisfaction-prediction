
from flask import Flask, render_template, request
import pandas as pd

from src.pipeline.predict_pipeline import PredictionPipeline

app = Flask(__name__)

predictor = PredictionPipeline()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read submitted passenger details
        passenger_data = {
            "Gender": request.form["Gender"],
            "Customer Type": request.form["Customer Type"],
            "Age": float(request.form["Age"]),
            "Type of Travel": request.form["Type of Travel"],
            "Class": request.form["Class"],
            "Flight Distance": float(request.form["Flight Distance"]),
        }

        # Remaining service ratings
        rating_columns = [
            "Inflight wifi service",
            "Departure/Arrival time convenient",
            "Ease of Online booking",
            "Gate location",
            "Food and drink",
            "Online boarding",
            "Seat comfort",
            "Inflight entertainment",
            "On-board service",
            "Leg room service",
            "Baggage handling",
            "Checkin service",
            "Inflight service",
            "Cleanliness",
        ]

        for col in rating_columns:
            passenger_data[col] = int(request.form[col])

        passenger_data["Departure Delay in Minutes"] = float(
            request.form["Departure Delay in Minutes"]
        )

        arrival_delay = request.form["Arrival Delay in Minutes"].strip()
        passenger_data["Arrival Delay in Minutes"] = (
            float(arrival_delay) if arrival_delay else float("nan")
        )

        passenger_df = pd.DataFrame([passenger_data])

        # Get blended probability
        probability = float(predictor.predict(passenger_df)[0])

        result = (
            "Satisfied"
            if probability >= 0.5
            else "Neutral or Dissatisfied"
        )

        return render_template(
            "index.html",
            prediction=result,
            probability=f"{probability * 100:.2f}%",
        )

    except (ValueError, KeyError) as error:
        return render_template(
            "index.html",
            error=f"Please check your input values: {error}",
        ), 400


if __name__ == "__main__":
    app.run(debug=True)