"""
Prediction module for new data
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# =========================
# PATH CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "model"


# =========================
# PREDICTOR CLASS
# =========================
class DiabetesPredictor:
    """
    Class to make predictions on new patients
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer = None

        self.feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]

    # =========================
    # LOAD MODELS
    # =========================
    def load_model(self):
        self.model = joblib.load(MODEL_DIR / "model.pkl")
        print("Model loaded")

    def load_scaler(self):
        self.scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        print("Scaler loaded")

    def load_imputer(self):
        self.imputer = joblib.load(MODEL_DIR / "imputer.pkl")
        print("Imputer loaded")

    # =========================
    # INTERPRETATION
    # =========================
    def interpret_result(self, prediction, probability):
        if probability >= 0.7:
            return f"High risk ({probability:.2%})"
        elif probability >= 0.4:
            return f"Moderate risk ({probability:.2%})"
        else:
            return f"Low risk ({probability:.2%})"

    # =========================
    # PREPROCESSING
    # =========================
    def preprocess_input(self, input_data):

        if self.scaler is None or self.imputer is None:
            raise ValueError("Scaler or Imputer not loaded.")

        # dict or dataframe
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()

        # check columns
        for col in self.feature_names:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        # handle zeros
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        df[zero_cols] = df[zero_cols].replace(0, np.nan)

        # imputation
        df_imputed = pd.DataFrame(
            self.imputer.transform(df),
            columns=df.columns
        )

        # scaling
        df_scaled = pd.DataFrame(
            self.scaler.transform(df_imputed),
            columns=df_imputed.columns
        )

        return df_scaled

    # =========================
    # SINGLE PREDICTION
    # =========================
    def predict(self, input_data):

        if self.model is None:
            raise ValueError("Model not loaded.")

        X = self.preprocess_input(input_data)

        probability = self.model.predict_proba(X)[0][1]

        #  optimized threshold
        threshold = 0.3
        prediction = int(probability >= threshold)

        return prediction, probability

    # =========================
    # BATCH PREDICTION
    # =========================
    def predict_batch(self, input_data):

        if self.model is None:
            raise ValueError("Model not loaded.")

        X = self.preprocess_input(input_data)

        probabilities = self.model.predict_proba(X)[:, 1]

        threshold = 0.3
        predictions = (probabilities >= threshold).astype(int)

        return predictions, probabilities


# =========================
# EXAMPLE USAGE
# =========================
def predict_single_patient():

    print("=" * 60)
    print("PREDICTION FOR NEW PATIENT")
    print("=" * 60)

    predictor = DiabetesPredictor()

    predictor.load_model()
    predictor.load_scaler()
    predictor.load_imputer()

    patient_data = {
        'Pregnancies': 2,
        'Glucose': 140,
        'BloodPressure': 80,
        'SkinThickness': 25,
        'Insulin': 100,
        'BMI': 32.5,
        'DiabetesPedigreeFunction': 0.5,
        'Age': 35
    }

    print("\nPatient data:")
    for k, v in patient_data.items():
        print(f"  {k}: {v}")

    prediction, probability = predictor.predict(patient_data)

    interpretation = predictor.interpret_result(prediction, probability)

    print("\n" + "=" * 60)
    print("\nRESULT")
    print("=" * 60)

    print(f"Prediction: {'DIABETIC' if prediction == 1 else 'NON-DIABETIC'}")
    print(f"Probability: {probability:.2%}")
    print(f"Interpretation: {interpretation}")
    print("=" * 60)

    return prediction, probability


# =========================
# RUN
# =========================
if __name__ == "__main__":
    predict_single_patient()