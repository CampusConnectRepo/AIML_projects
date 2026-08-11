"""Predict the species of an iris flower from its measurements."""

import os
import sys

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



def predict(model, target_names, measurements):
    """Predict the species for one flower given its 4 measurements (cm)."""
    probabilities = model.predict_proba([measurements])[0]
    index = probabilities.argmax()

    print(f"\nMeasurements: {measurements}")
    print(f"Prediction: {target_names[index]} ({probabilities[index]:.1%} confidence)")


def main():

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Test accuracy: {accuracy:.2%}")    

    #Save the trained model and target names
    #model, target_names = train_model()
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/iris_model.pkl")

    #save a tiny metrics file 
    acc = model.score(X_test, y_test)
    with open("artifacts/metrics.txt", "w") as f:
        f.write(f"Test Accuracy: {acc:.2%}\n")




if __name__ == "__main__":
    main()
