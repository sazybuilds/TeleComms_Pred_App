import requests

# fastapi endpoint url
API_URL = "http://127.0.0.1:8000/predict"

payload = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 55.50,
    "TotalCharges": 666.00
}

# send POST request
response = requests.post(API_URL, json=payload)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
