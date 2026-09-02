# pyrefly: ignore [missing-import]
import streamlit as st
import requests
import os
from pathlib import Path
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv


# Load environment variables
load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
API_KEY = os.getenv("API_KEY", "")

st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp {
        background-color: #f0f4f8;
        color: #1a1a2e;
    }
    p, label, span, div {
        color: #1a1a2e;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1f77b4, #0b5394);
        color: #ffffff !important;
        border-radius: 12px;
        height: 58px;
        font-size: 20px;
        font-weight: 700;
        letter-spacing: 0.5px;
        border: none;
        transition: all 0.25s ease;
        box-shadow: 0 4px 14px rgba(31, 119, 180, 0.35);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0b5394, #073763);
        box-shadow: 0 6px 20px rgba(11, 83, 148, 0.45);
        transform: translateY(-2px);
    }
    .stRadio {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 8px 12px;
        margin-bottom: 8px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
    }
    .stRadio > label {
        color: #1a1a2e !important;
        font-weight: 600;
        font-size: 14px;
    }
    .stRadio > div * {
        color: #1a1a2e !important;
    }
    h3 {
        color: #0b5394 !important;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }
    .stSelectbox > label,
    .stNumberInput > label,
    .stSlider > label {
        color: #1a1a2e !important;
        font-weight: 600;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE NAVIGATION STATE
# ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to_predict():
    st.session_state.page = "predict"

def go_to_home():
    st.session_state.page = "home"

# ─────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────
if st.session_state.page == "home":
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        # ── Profile photo ────────────────────────────────────────────────────
        photo_path = Path(__file__).parent / "profile.jpg"
        if photo_path.exists():
            import base64
            img_b64 = base64.b64encode(photo_path.read_bytes()).decode()
            st.markdown(
                f"""
                <div style="display:flex; justify-content:center; margin-bottom:16px;">
                    <img src="data:image/jpeg;base64,{img_b64}"
                         style="width:160px; height:160px; object-fit:cover;
                                border-radius:50%;
                                border:4px solid #1f77b4;
                                box-shadow:0 4px 20px rgba(31,119,180,0.35);
                                pointer-events:none;
                                user-select:none;" />
                </div>
                """,
                unsafe_allow_html=True
            )
        # ── Heading & author info ─────────────────────────────────────────────
        st.title("📡 Telco Customer Churn Predictor")
        st.markdown("### A telecommunications machine learning application that predicts whether a customer is likely to churn.")
        st.markdown("---")
        st.markdown("""
        **Author:** Osazuwa Oaikhina  
        **LinkedIn:** [Connect with me!](https://www.linkedin.com/in/osazuwaoaikhina-5937ab268/)  
        **GitHub:** [View my projects](https://github.com/sazybuilds)
        """)
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.button("🚀 Enter Predictor", on_click=go_to_predict, use_container_width=True)



elif st.session_state.page == "predict":
    st.button("⬅️ Back to Home", on_click=go_to_home)


    st.title("📡 TeleComms Customer Churn Predictor")
    st.markdown("##### Fill in the customer's profile below, then click **Predict**.")
    st.divider()

    col_demographics, col_services, col_billing = st.columns(3, gap="large")
    
    with col_demographics:
        st.subheader("👤 Demographics")
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
        senior_citizen = st.radio(
            "Senior Citizen", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            horizontal=True
        )
        partner = st.radio("Partner", ["Yes", "No"], horizontal=True)
        dependents = st.radio("Dependents", ["No", "Yes"], horizontal=True)
    
    with col_services:
        st.subheader("🔌 Services")
        phone_service = st.radio("Phone Service", ["No", "Yes"], horizontal=True)
        multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        internet_service = st.radio("Internet Service", ["DSL", "Fiber optic", "No"], horizontal=True)
        
        with st.expander("🌐 Additional Internet Services", expanded=True):
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    
    with col_billing:
        st.subheader("💳 Billing & Account")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.radio("Paperless Billing", ["Yes", "No"], horizontal=True)
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    tenure = st.slider("Tenure (months)", min_value=0, max_value=120, value=1)
    
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        monthly_charges = st.number_input("Monthly ($)", min_value=0.0, value=50.0, step=1.0)
    with sub_col2:
        total_charges = st.number_input("Total ($)", min_value=0.0, value=50.0, step=5.0)

    st.markdown("<br>", unsafe_allow_html=True)
    _, center_col, _ = st.columns([1, 2, 1])

    with center_col:
        predict_clicked = st.button("🚀 Predict Customer Churn", use_container_width=True)

    if predict_clicked:
        with st.spinner("🔍 Analyzing customer data..."):
            input_data = {
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges
            }

            # Send the data to the FastAPI backend
            try:
                response = requests.post(
                    API_URL,
                    json=input_data,
                    headers={"X-API-Key": API_KEY}
                )
                if response.status_code == 200:
                    result = response.json()
                    prediction = result["prediction"]
                    probability = result["churn_probability"]
                elif response.status_code == 401:
                    st.error("Authentication error: the API key is missing or incorrect. "
                             "Please check your API_KEY environment variable.")
                    prediction = None
                    probability = None
                elif response.status_code == 400:
                    detail = response.json().get("detail", "Invalid input data.")
                    st.error(f"Invalid input: {detail}")
                    prediction = None
                    probability = None
                elif response.status_code == 429:
                    st.warning("Too many requests. Please wait a moment before trying again.")
                    prediction = None
                    probability = None
                else:
                    st.error(f"Unexpected error from API (HTTP {response.status_code}).")
                    prediction = None
                    probability = None
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot connect to API. Please ensure the backend server is running on {API_URL}.")
                prediction = None
                probability = None


        st.divider()
        if prediction is not None:
            _, result_col, _ = st.columns([1, 2, 1])
            with result_col:
                if prediction == 1:
                    st.error(f"### ■ High Risk\nThis customer is likely to **CHURN**" 
                                f"({probability:.1%} probability).")
                else:
                   st.success(f"### ■ Retained\nThis customer is likely to **STAY** "
                                f"({(1 - probability):.1%} probability of staying")
