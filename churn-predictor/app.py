from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts the probability that a telecom customer will churn, "
                 "based on account and service usage features.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(MODEL_PATH)


class CustomerFeatures(BaseModel):
    tenure: int = Field(..., ge=0, le=100, description="Months as a customer")
    monthly_charges: float = Field(..., ge=0, description="Monthly bill amount")
    total_charges: float = Field(..., ge=0, description="Total amount billed so far")
    contract: str = Field(..., description="Month-to-month | One year | Two year")
    internet_service: str = Field(..., description="DSL | Fiber optic | No")
    tech_support: str = Field(..., description="Yes | No")
    online_security: str = Field(..., description="Yes | No")
    paperless_billing: str = Field(..., description="Yes | No")
    payment_method: str = Field(
        ..., description="Electronic check | Mailed check | Bank transfer | Credit card"
    )
    num_support_calls: int = Field(..., ge=0, le=20)


@app.get("/")
def root():
    return {"message": "Churn Prediction API is running. See /docs for the interactive API, or open /app for the demo UI."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: CustomerFeatures):
    try:
        df = pd.DataFrame([features.model_dump()])
        proba = model.predict_proba(df)[0][1]
        prediction = int(proba >= 0.5)
        risk_level = "High" if proba >= 0.66 else "Medium" if proba >= 0.33 else "Low"
        return {
            "churn_prediction": prediction,
            "churn_probability": round(float(proba), 4),
            "risk_level": risk_level,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Serve the simple frontend at /app
app.mount("/app", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="static")
