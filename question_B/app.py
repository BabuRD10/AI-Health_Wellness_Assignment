from flask import Flask, request, render_template
import pandas as pd
import joblib
from pathlib import Path


app = Flask(__name__)


# Find the main project folder
BASE_DIR = Path(__file__).resolve().parent.parent


# Load the trained Random Forest model
MODEL_PATH = BASE_DIR / "question_B" / "diabetes_model.pkl"
model = joblib.load(MODEL_PATH)


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():

    try:
        pregnancies = float(request.form["Pregnancies"])
        glucose = float(request.form["Glucose"])
        blood_pressure = float(request.form["BloodPressure"])
        skin_thickness = float(request.form["SkinThickness"])
        insulin = float(request.form["Insulin"])
        bmi = float(request.form["BMI"])
        diabetes_pedigree = float(
            request.form["DiabetesPedigreeFunction"]
        )
        age = float(request.form["Age"])

        features = pd.DataFrame([[
           pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]], columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ])


        prediction = model.predict(features)[0]

        if prediction == 1:
            result = "Higher predicted diabetes risk"
        else:
            result = "Lower predicted diabetes risk"

        return render_template(
            "index.html",
            result=result
        )

    except Exception as error:
        return render_template(
            "index.html",
            result="Invalid input. Please enter valid numbers."
        )


if __name__ == "__main__":
    app.run(debug=True)