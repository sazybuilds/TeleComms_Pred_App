"""
tune.py — Hyperparameter Tuning Script
=======================================
Runs a GridSearchCV over a LogisticRegression pipeline to find the best
solver / regularisation combination, then prints the best score and params.

This script was extracted from the original EDA notebook
(ml_scripts/telco_customer_churn.ipynb) — Section 7: Hyperparameter Tuning.

Run from the project root:
    python ml_scripts/tune.py
"""

import os
import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier

# Ensure the project root is in sys.path for absolute imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from ml_scripts import func as prep_func


def tune():
    load_dotenv()

    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
    DATASET_PATH = PROJECT_ROOT / os.getenv("DATASET_NAME")
    LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR") / os.getenv("LOG_NAME")

    TARGET_COL = os.getenv("TARGET_COL")
    TEST_SIZE = float(os.getenv("TEST_SIZE"))
    RANDOM_STATE = int(os.getenv("RANDOM_STATE"))

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_PATH),
        ],
    )

    logging.info("Tuning script started")
    df = pd.read_csv(DATASET_PATH)
    logging.info(f"Dataset loaded — shape: {df.shape}")

    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    binary_features, nominal_features, numeric_features = [], [], []

    for col in X_train.columns.tolist():
        n = X_train[col].nunique()
        if n == 2:
            binary_features.append(col)
        elif 2 < n < 10:
            nominal_features.append(col)
        else:
            numeric_features.append(col)

    binary_features.remove("SeniorCitizen")

    COL_WITH_ZEROS = ["TotalCharges"]


    modify_str_int_column = FunctionTransformer(
        prep_func.modify_total_charges,
        kw_args={"cols": COL_WITH_ZEROS},
        validate=False
    )
    zero_to_nan = FunctionTransformer(
        prep_func.replace_zeros_with_nan_df,
        kw_args={"cols": COL_WITH_ZEROS},
        validate=False,
    )
    encode_binary = FunctionTransformer(
        prep_func.encode_binary_cols,
        kw_args={"cols": binary_features},
        validate=False,
    )

    binary_transformer = Pipeline(steps=[("encode", encode_binary)])
    nominal_transformer = Pipeline(steps=[("oneHot", OneHotEncoder())])

    numerical_transformer = Pipeline(
        steps=[
            ("modify_odd_column", modify_str_int_column),
            ("zero_to_nan", zero_to_nan),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("binary", binary_transformer, binary_features),
            ("nominal", nominal_transformer, nominal_features),
            ("numerical", numerical_transformer, numeric_features),
        ]
    )

    logging.info("Preprocessing pipeline built")

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / max(pos, 1)

    k = 10
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "LogisticRegression": LogisticRegression(
            random_state=RANDOM_STATE, class_weight="balanced", max_iter=1000
        ),
        "RandomForest": RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight="balanced_subsample"
        ),
        "HistGB": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "SVC": SVC(random_state=RANDOM_STATE, class_weight="balanced"),
        "DecisionTree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "XGB": XGBClassifier(
            scale_pos_weight=scale_pos_weight, eval_metric="logloss"
        ),
        "NaiveBayes": BernoulliNB(
            alpha=1.0,      # Laplace smoothing
            binarize=0.0,   # Threshold for binarising continuous inputs
            fit_prior=True, # Account for class imbalance via prior probs
        ),
    }

    scoring = {
        "accuracy": "accuracy",
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
    }

    logging.info("Running cross-validated model comparison …")
    rows = []
    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append(
            {
                "model": name,
                "accuracy": scores["test_accuracy"].mean(),
                "recall": scores["test_recall"].mean(),
                "precision": scores["test_precision"].mean(),
                "f1_score": scores["test_f1"].mean(),
            }
        )

    cv_results = pd.DataFrame(rows).sort_values("f1_score", ascending=False)
    best_row = cv_results.iloc[0]
    best_model = best_row["model"]
    print("\n===== Cross-Validation Results (sorted by F1) =====")
    print(cv_results.to_string(index=False))
    logging.info("Cross-validation complete")
    logging.info(f"Best Model {best_model}")



    logging.info("Starting GridSearchCV …")

    best_model_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    max_iter=1000,
                ),
            ),
        ]
    )

    param_grid = [
        # Grid 1: Pure L1 (l1_ratio=1) and pure L2 (l1_ratio=0) — liblinear & saga
        {
            "model__solver": ["liblinear", "saga"],
            "model__l1_ratio": [0, 1],
            "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "model__max_iter": [1000],
        },
        # Grid 2: Pure L2 only — lbfgs and newton-cg (no L1 support)
        {
            "model__solver": ["lbfgs", "newton-cg"],
            "model__l1_ratio": [0],
            "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "model__max_iter": [1000],
        },
        # Grid 3: ElasticNet (mix of L1+L2) — requires saga solver
        {
            "model__solver": ["saga"],
            "model__l1_ratio": [0.2, 0.5, 0.8],
            "model__C": [0.001, 0.01, 0.1, 1, 10],
            "model__max_iter": [2000],  # Higher iterations for ElasticNet convergence
        },
    ]

    grid = GridSearchCV(
        estimator=best_model_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=2,
    )

    grid.fit(X_train, y_train)

    print("\n===== GridSearchCV Results =====")
    print(f"Best F1 Score : {grid.best_score_:.4f}")
    print(f"Best Params   : {grid.best_params_}")

    logging.info(f"Best F1 score : {grid.best_score_:.4f}")
    logging.info(f"Best params   : {grid.best_params_}")

    logging.info("Tuning script finished")
    logging.info("Best Parameters and Model gotten")


if __name__ == "__main__":
    tune()
