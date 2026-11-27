import math
import os
from datetime import date

import requests
from flask import Flask, render_template, request

app = Flask(__name__)

API_URL = os.getenv("API_URL", "http://model-api:8000")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            prediction_date = date.fromisoformat(
                request.form["prediction_date"]
            )

            lag_1 = int(request.form["lag_1"])
            lag_7 = int(request.form["lag_7"])
            rolling_mean_7 = float(request.form["rolling_mean_7"])

            if (
                min(lag_1, lag_7, rolling_mean_7) < 0
                or not math.isfinite(rolling_mean_7)
            ):
                raise ValueError("Invalid crime counts")

            features = {
                "day_of_week": prediction_date.weekday(),
                "month": prediction_date.month,
                "is_weekend": int(prediction_date.weekday() >= 5),
                "lag_1": lag_1,
                "lag_7": lag_7,
                "rolling_mean_7": rolling_mean_7,
            }

            response = requests.post(
                f"{API_URL}/predict",
                json={"features": features},
                timeout=60,
            )

            if response.status_code == 503:
                error = "The trained model is currently unavailable."
            else:
                response.raise_for_status()
                prediction = round(
                    float(response.json()["prediction"]), 1
                )

        except (ValueError, KeyError):
            error = "Please enter valid dates and crime counts."

        except requests.exceptions.RequestException:
            error = "Unable to retrieve a prediction from the model API."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        values=request.form,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
