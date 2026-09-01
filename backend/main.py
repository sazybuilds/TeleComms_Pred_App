# pyrefly: ignore [missing-import]
from fastapi import FastAPI
# pyrefly: ignore [missing-import]
from pydantic import BaseModel

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ml_scripts.prediction import predict

app = FastAPI(title="ML model API")

class PredictionInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
@app.post("/predict")
def predict_churn(customer_data: PredictionInput):
    # Convert the pydantic model to a dictionary
    input_dict = customer_data.model_dump()
    
    # Get prediction
    prediction = predict(input_data=input_dict)
    
    # Ensure prediction is a native int/float before returning
    return {"prediction": int(prediction)}


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Telco Customer Churn API",
        "description": "A machine learning API to predict customer churn in the telecommunications industry.",
        "author": "Osazuwa Oaikhina",
        "linkedin": "https://www.linkedin.com/in/osazuwaoaikhina-5937ab268/"
    }