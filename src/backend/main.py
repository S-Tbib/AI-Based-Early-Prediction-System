import sys
import os

DEBUG = os.getenv("DEBUG", "False").lower() == "true"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")
SCALER_PATH = os.getenv("SCALER_PATH")
IMPUTER_PATH = os.getenv("IMPUTER_PATH")

from pydantic import BaseModel
from models.predict import DiabetesPredictor
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AI Diabetes Prediction API",
    description="API for diabetes risk prediction",
    version="1.0.0"
)

# CORS MUST be after app creation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
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


@app.get("/model-info")
def get_model_info():
    """Get detailed model information including feature importances"""
    try:
        model_info = predictor.get_model_info()
        return {
            "status": "success",
            "model_info": model_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    