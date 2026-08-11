import argparse 
import json
from pathlib import Path
import numpy as np
import joblib


def load_model():
    """Load the trained model and target names from disk."""
    model_path = Path("MLOPS/artifacts/iris_model.pkl")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    return model

def main():
    parse = argparse.ArgumentParser(description="Run the trained model on new data.")
    parse.add_argument(
        "--input", type=str, required=True, help="Path to the input JSON file with flower measurements."
    )
    args = parse.parse_args()

    # Load the model
    model = load_model()

    # Read the input data
    with open(args.input, "r") as f:
        data = json.load(f)

    # Make a prediction
    measurements = [data["sepal_length"], data["sepal_width"], data["petal_length"], data["petal_width"]]
    prediction = model.predict([measurements])[0]

    print(f"Predicted species: {prediction}")