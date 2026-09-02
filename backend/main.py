# pyrefly: ignore [missing-import]
from fastapi import FastAPI, Request, Security, HTTPException, status
import os
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.responses import JSONResponse
# pyrefly: ignore [missing-import]
from fastapi.security import APIKeyHeader
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
# pyrefly: ignore [missing-import]
from slowapi import Limiter, _rate_limit_exceeded_handler
# pyrefly: ignore [missing-import]
from slowapi.util import get_remote_address
# pyrefly: ignore [missing-import]
from slowapi.errors import RateLimitExceeded

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ml_scripts.prediction import predict

# ── API Key authentication ───────────────────────────────────────────────────
# The key is stored as API_KEY in the environment (HF Secret / Streamlit Secret).
# Every /predict request must include the header:  X-API-Key: <your-key>
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(_api_key_header)) -> None:
    """FastAPI dependency — rejects requests with a missing or wrong API key."""
    expected = os.getenv("API_KEY")
    if not expected:
        # Fail-open only if the env var was never set (local dev without .env).
        # In production the key will always be present, so this branch won't run.
        return
    if api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Set the X-API-Key header.",
        )

# ── Rate limiter (slowapi) ────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Telco Churn Prediction API")

# ── CORS ─────────────────────────────────────────────────────────────────────
# Allows the Streamlit Community Cloud frontend (different domain) to call this
# API from the browser without being blocked by the same-origin policy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://osaz-telco-churn.streamlit.app/"],   # tighten to your Streamlit URL in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Attach rate-limit exceeded handler so slowapi returns a proper 429 JSON body
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Telco Customer Churn API",
        "description": "A machine learning API to predict customer churn in the telecommunications industry.",
        "author": "Osazuwa Oaikhina",
        "linkedin": "https://www.linkedin.com/in/osazuwaoaikhina-5937ab268/"
    }

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


@app.post("/predict", dependencies=[Security(verify_api_key)])
@limiter.limit("30/minute")
def predict_churn(request: Request, customer_data: PredictionInput):
    """Predict customer churn.
    Rate-limited to 30 requests per minute per IP address.
    Returns HTTP 400 for unknown categorical values the model was not trained on.
    """
    input_dict = customer_data.model_dump()
    try:
        result = predict(input_data=input_dict)
    except ValueError as exc:
        # The OneHotEncoder raises ValueError when it encounters a category it
        # was not fitted on (e.g. InternetService="Satellite").  Return a
        # descriptive 400 instead of leaking a bare 500 to the caller.
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_category",
                "detail": (
                    f"One or more field values were not seen during model training: {exc}. "
                    "Please use only the supported dropdown values."
                )
            }
        )
    return {
        "prediction": int(result["prediction"]),
        "churn_probability": result["churn_probability"]
    }



