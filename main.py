from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# =========================================================
# Project paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "credit_risk_model.pkl"
THRESHOLD_PATH = BASE_DIR / "best_threshold.pkl"


# =========================================================
# Load ML model
# =========================================================

print("Loading credit risk model...")

model = joblib.load(MODEL_PATH)
threshold = float(joblib.load(THRESHOLD_PATH))

print("Model loaded successfully")
print(f"Threshold: {threshold}")


# =========================================================
# FastAPI application
# =========================================================

app = FastAPI(
    title="Credit Risk Assessment API",
    description="XGBoost-based credit default risk prediction API.",
    version="1.0.0",
)


# =========================================================
# Request schema
# =========================================================

class LoanApplication(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int


# =========================================================
# Health check
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "threshold": threshold,
    }


# =========================================================
# Prediction endpoint
# =========================================================

@app.post("/predict")
def predict(data: LoanApplication):

    # Convert request data into DataFrame
    input_df = pd.DataFrame([data.model_dump()])

    print("Received prediction request:")
    print(input_df)

    # Get probability of default
    probability = model.predict_proba(input_df)[0][1]

    # Convert NumPy value to Python float
    probability = float(probability)

    # Make binary prediction
    prediction = int(probability >= threshold)

    print(f"Probability: {probability}")
    print(f"Threshold: {threshold}")
    print(f"Prediction: {prediction}")

    return {
        "default_probability": probability,
        "default_prediction": prediction,
        "threshold": threshold,
        "Result": "High Risk" if prediction == 1 else "Low Risk",
    }


# =========================================================
# Frontend
# =========================================================

@app.get("/")
def home():
    return FileResponse(
        BASE_DIR / "static" / "index.html"
    )


# Serve CSS, JavaScript and other static files
app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)