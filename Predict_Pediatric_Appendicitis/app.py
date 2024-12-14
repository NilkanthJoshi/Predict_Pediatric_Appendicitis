from flask import Flask, request, render_template, jsonify, send_file
from src.pipeline.train_pipeline import TrainingPipeline
from src.pipeline.predict_pipeline import PredictionPipeline
import os

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/train", methods=["GET"])
def train_route():
    train_pipeline = TrainingPipeline()
    train_pipeline.run_pipeline()
    return render_template("home.html", message="Training completed successfully!")


@app.route("/predict", methods=["POST", "GET"])
def predict_route():
    if request.method == "POST":
        prediction_pipeline = PredictionPipeline(request)
        prediction_file_details = prediction_pipeline.run_pipeline()
        return send_file(
            prediction_file_details.prediction_file_path,
            download_name=prediction_file_details.prediction_file_name,
            as_attachment=True,
        )
    else:
        return render_template("upload_file.html")


if __name__ == "__main__":
    app.run(port=5000, debug=True)
