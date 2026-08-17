from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import InputValidationError, load_artifacts, predict_payload

app = Flask(__name__)
model, metadata = load_artifacts()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/intro")
def intro():
    return render_template("intro.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")
    payload = {
        "Temp": request.form.get("temp"), "Humidity": request.form.get("humidity"),
        "Cloud Cover": request.form.get("cloud_cover"), "Jan-Feb": request.form.get("rain_janfeb"),
        "Mar-May": request.form.get("rain_mar_may"), "Oct-Dec": request.form.get("rain_octdec"),
        "avgjune": request.form.get("avg_june"), "sub": request.form.get("sub"),
    }
    try:
        result = predict_payload(model, payload)
    except InputValidationError as exc:
        return render_template("predict.html", error=str(exc), values=request.form), 400
    percent = round(result["probability"] * 100, 2)
    template = "high.html" if result["prediction"] == 1 else "low.html"
    return render_template(template, percent=percent, model_name=metadata["model_name"])


if __name__ == "__main__":
    app.run(debug=False)
