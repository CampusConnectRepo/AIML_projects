from flask import Flask, request, jsonify
import joblib
from pathlib import Path

app = Flask(__name__)
Model_PATH = Path(__file__).resolve().parent / "artifacts" / "iris_model.pkl"

if not Model_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {Model_PATH}")

model = joblib.load(Model_PATH)

@app.route("/")
def home():
    return "Welcome to the Iris Flower Prediction API! Use the /predict endpoint to get predictions."   

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Invalid input. Please provide 'features' in the request body."}), 400
    features = data["features"]
    if len(features) != 4:
        return jsonify({"error": "Invalid input. 'features' must be a list of 4 measurements."}), 400
    prediction = model.predict([features])
    return jsonify({"prediction": prediction.tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)