from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Model paths
MODEL_PATH = BASE_DIR / "credit_risk_model.pkl"
THRESHOLD_PATH = BASE_DIR / "best_threshold.pkl"

# ML model storage
ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and threshold when the application starts
    ml_model["model"] = joblib.load(MODEL_PATH)
    ml_model["threshold"] = float(joblib.load(THRESHOLD_PATH))

    print("Model loaded successfully")
    print(f"Threshold: {ml_model['threshold']}")

    yield

    # Clear model when application shuts down
    ml_model.clear()


app = FastAPI(
    title="Credit Risk Assessment API",
    description="XGBoost-based credit default risk prediction API.",
    version="1.0.0",
)


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


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": "model" in ml_model,
        "threshold": ml_model.get("threshold"),
    }


@app.post("/predict")
def predict(data: LoanApplication):

    # Convert request data into DataFrame
    input_df = pd.DataFrame([data.model_dump()])

    # Get probability of default
    probability = ml_model["model"].predict_proba(input_df)[0][1]

    # Convert NumPy value to Python float
    probability = float(probability)

    # Convert threshold to Python float
    threshold = float(ml_model["threshold"])

    # Make binary prediction
    prediction = int(probability >= threshold)

    return {
        "default_probability": probability,
        "default_prediction": prediction,
        "threshold": threshold,
        "Result": "High Risk" if prediction == 1 else "Low Risk",
    }


# Serve frontend
@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


# Serve CSS, JavaScript and other static assets
app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)