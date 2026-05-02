import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models.predict import DiabetesPredictor


# =========================
# INIT APP
# =========================
app = FastAPI(
    title="AI Diabetes Prediction API",
    description="API for diabetes risk prediction",
    version="1.0.0"
)

print("API started successfully")


# =========================
# LOAD MODEL
# =========================
predictor = DiabetesPredictor()
predictor.load_model()
predictor.load_scaler()
predictor.load_imputer()


# =========================
# SCHEMA
# =========================
class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


# =========================
# ROUTES
# =========================
@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Diabetes Prediction API is running",
        "docs": "/docs"
    }


@app.post("/predict")
def predict_diabetes(patient: PatientData):
    try:
        data = patient.dict()

        prediction, probability = predictor.predict(data)
        interpretation = predictor.interpret_result(prediction, probability)

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "risk": interpretation,
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None
    }