<div align="center">

# 📡 Telco Customer Churn Predictor

**An end-to-end machine learning system that predicts whether a telecom customer will churn — built from raw data exploration all the way to a live REST API and interactive web interface.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.62-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io)

</div>

---

## 🧠 What This Project Does

> Given a telecom customer's demographics, services, and billing information, the model predicts whether that customer is likely to **churn** (leave) or **stay**.

The project covers the **complete ML development lifecycle**:

```
Jupyter Notebook  →  Modular Scripts  →  REST API  →  Interactive Web UI
     (EDA)           (train / tune)      (FastAPI)      (Streamlit)
```

---

## 🗂️ Project Structure

```
Telco_cust_churn/
│
├── 🖥️  backend/
│   └── main.py                     # FastAPI app — exposes /predict endpoint
│
├── 🎨  frontend/
│   └── demo_app.py                 # Streamlit UI — talks to the API over HTTP
│
├── 🤖  ml_scripts/
│   ├── telco_customer_churn.ipynb  # Original EDA & experimentation notebook
│   ├── func.py                     # Custom preprocessing helpers
│   ├── training.py                 # Standalone model training script
│   ├── tune.py                     # Model comparison + GridSearchCV tuning
│   └── prediction.py               # Inference script — loads model, runs predictions
│
├── 📁  model_dir/                  # Saved .joblib pipeline  (git-ignored)
├── 📋  logs/                       # Runtime logs            (git-ignored)
├── 📊  dataset/                    # Raw CSV dataset         (git-ignored)
│
├── env_template.txt                # Template for .env variables
└── requirements.txt                # Project dependencies
```

---

## 📊 Dataset

This project uses the [**Telco Customer Churn dataset**](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle.

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` and configure its path in your `.env` file.

---

## ⚙️ Setup

### 1 · Environment Variables
```bash
cp env_template.txt .env
# Then open .env and fill in your local paths
```
> ⚠️ `.env` is already in `.gitignore` — never commit secrets!

### 2 · Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3 · Install Dependencies
```bash
pip install -r requirements.txt
```

> **🗒️ Notebook users:** `ipykernel` is intentionally excluded from `requirements.txt` — it's a dev-only tool for running `.ipynb` files in your editor and is not needed to run the app. Install it separately:
> ```bash
> pip install ipykernel
> ```

---

## 🚀 Running the Project

| Step | Command | What it does |
|------|---------|--------------|
| **Train** | `python ml_scripts/training.py` | Preprocesses data, trains & saves the model pipeline |
| **Tune** | `python ml_scripts/tune.py` | Cross-validates 7 models, runs GridSearchCV, prints best params |
| **API** | `uvicorn backend.main:app --reload` | Starts FastAPI server at `http://localhost:8000` |
| **UI** | `streamlit run frontend/demo_app.py` | Launches the Streamlit prediction interface |

> **💡 Tip:** The API must be running before the Streamlit UI can make predictions. Start the API first.
>
> FastAPI's interactive docs are auto-generated at: `http://localhost:8000/docs`

---

## 🔬 Model Details

| Property | Value |
|----------|-------|
| **Algorithm** | Logistic Regression |
| **Solver** | `saga` |
| **Penalty** | ElasticNet (`l1_ratio=0.8`) |
| **Regularisation (C)** | `1` |
| **Class weighting** | `balanced` |
| **Test Accuracy** | ~73–75% |
| **Serialisation** | `joblib.dump()` — full sklearn Pipeline |

**Preprocessing pipeline:**

```
Binary features   ──►  Custom label encoding (FunctionTransformer)
Nominal features  ──►  OneHotEncoder
Numerical features ──► Zero→NaN  ──►  Median Imputation  ──►  StandardScaler
                                        ↑
                           (TotalCharges had blank strings, not NaN)
```

---

## 🗺️ ML Development Journey

<details>
<summary><b>Phase 1 — 📓 Exploration & EDA (Notebook)</b></summary>

> `ml_scripts/telco_customer_churn.ipynb`

All initial work was done in a Jupyter notebook to allow fast, interactive iteration:

- **EDA**: Loaded the dataset, inspected class distributions, checked dtypes. Discovered that `TotalCharges` stored spaces `" "` instead of `NaN` — converted with `pd.to_numeric(..., errors='coerce')`.
- **Feature categorisation**: Dynamically grouped all 19 features into **binary**, **nominal**, and **numerical** based on unique value counts (no hardcoding).
- **Preprocessing design**: Built a `ColumnTransformer` applying a different `Pipeline` to each group — custom encoding for binary, `OneHotEncoder` for nominal, and imputation + scaling for numerical.
- **Baseline**: Trained a plain `LogisticRegression` pipeline as the reference point.

</details>

<details>
<summary><b>Phase 2 — ⚖️ Model Selection (Cross-Validation)</b></summary>

> `ml_scripts/tune.py`

Ran **10-fold StratifiedKFold cross-validation** across 7 candidate models, evaluating accuracy, recall, precision, and F1:

| Model | Strategy for class imbalance |
|-------|------------------------------|
| `LogisticRegression` | `class_weight='balanced'` |
| `RandomForestClassifier` | `class_weight='balanced_subsample'` |
| `HistGradientBoostingClassifier` | `class_weight='balanced'` |
| `SVC` | `class_weight='balanced'` |
| `DecisionTreeClassifier` | `class_weight='balanced'` |
| `XGBClassifier` | `scale_pos_weight = neg/pos` |
| `BernoulliNB` | `fit_prior=True` (Laplace smoothing) |

**✅ Winner**: Logistic Regression — best F1 score, balancing precision and recall on the imbalanced churn target.

</details>

<details>
<summary><b>Phase 3 — 🎯 Hyperparameter Tuning (GridSearchCV)</b></summary>

> `ml_scripts/tune.py`

Ran `GridSearchCV` (scored on **F1**, 10-fold CV) over 3 parameter grids covering all regularisation strategies supported by scikit-learn's Logistic Regression:

| Grid | Solvers | Regularisation |
|------|---------|----------------|
| Grid 1 | `liblinear`, `saga` | Pure L1 and pure L2 |
| Grid 2 | `lbfgs`, `newton-cg` | Pure L2 only (no L1 support) |
| Grid 3 | `saga` | **ElasticNet** (L1 + L2 mix, `l1_ratio ∈ {0.2, 0.5, 0.8}`) |

**🏆 Best parameters found:**
```
solver='saga'  |  l1_ratio=0.8  |  C=1  |  max_iter=2000
```

</details>

<details>
<summary><b>Phase 4 — 🔧 Notebook → Production Scripts</b></summary>

> `ml_scripts/training.py` · `ml_scripts/prediction.py` · `ml_scripts/func.py`

The notebook was refactored into three focused, standalone scripts:

- **`func.py`** — Extracted all custom helper functions (`encode_binary_cols`, `replace_zeros_with_nan_df`, `print_evaluation`). These *must* exist as a proper importable module so `joblib` can find them when deserialising the pipeline.
- **`training.py`** — Full training pipeline wrapped in `train_model()`. Fully environment-driven via `.env`, with structured logging to console and file.
- **`prediction.py`** — Loads the `.joblib` model once at **module level** (server startup) to avoid reloading on every request.

> **🔑 Key challenge solved:** After reorganising `func.py` into the `ml_scripts/` package, the model's pickle references still pointed to the old top-level `func` module name. Fixed with a `sys.modules` alias:
> ```python
> sys.modules['func'] = sys.modules['ml_scripts.func']
> ```

</details>

<details>
<summary><b>Phase 5 — ⚡ REST API (FastAPI)</b></summary>

> `backend/main.py`

Wrapped `predict()` in a FastAPI application:

- **`PredictionInput`** — a Pydantic model defining all 19 customer features with correct types. Gives free automatic request validation and a Swagger UI.
- **`/predict`** — POST endpoint that calls `.model_dump()` to convert the validated Pydantic object to a plain `dict`, passes it to `predict()`, and returns `{"prediction": int}`.
- `sys.path` is extended at startup to ensure `ml_scripts` is importable regardless of the working directory uvicorn is launched from.

</details>

<details>
<summary><b>Phase 6 — 🎨 Interactive Frontend (Streamlit)</b></summary>

> `frontend/demo_app.py`

Built a polished, multi-page Streamlit app:

- **Home page** — Project intro, author info, and a navigation button.
- **Predict page** — All 19 input fields organised into 3 columns (Demographics / Services / Billing). Clicking **Predict** POSTs the form data to the FastAPI backend and displays a colour-coded result card.
- **Fully decoupled** from the ML code — communicates only via HTTP, so the frontend and backend can be deployed independently.

</details>

<details>
<summary><b>Phase 7 — ✅ End-to-End Testing</b></summary>

Manual testing flow:

1. `uvicorn backend.main:app --reload` — start the API
2. `streamlit run frontend/demo_app.py` — launch the UI
3. Fill in a customer profile → verify the prediction card appears correctly
4. Hit `http://localhost:8000/docs` → test the `/predict` endpoint directly via Swagger UI

</details>

---

## 🛣️ Roadmap

- [x] Data exploration and model tuning
- [x] Automated training and prediction scripts
- [x] Build an interactive frontend using Streamlit
- [x] Serve the model via a REST API (FastAPI)
- [ ] PostgreSQL database integration

---

## 👤 Author

<div align="center">

**Osazuwa Oaikhina**  
*300 Level Computer Science Student — Covenant University*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/osazuwaoaikhina-5937ab268/)

</div>
