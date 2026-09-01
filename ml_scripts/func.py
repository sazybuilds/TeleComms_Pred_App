import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report
)


"""
No -----> 0
Yes ----> 1
"""

def replace_zeros_with_nan_df(X, cols):
    X = X.copy()
    for col in cols:
        if col in X.columns:
            X[col] = X[col].replace(0, np.nan)

    return X



def encode_binary_cols(X, cols):
    X = X.copy()
    for col in cols:
        if col in X.columns:
            X[col] = X[col].map({"No": 0, "Yes": 1, "Female": 0, "Male": 1})

    return X


def print_evaluation(title:str, ground_truth: np.ndarray, model_predictions: np.ndarray):
    acc = accuracy_score(ground_truth, model_predictions)
    class_report = classification_report(ground_truth, model_predictions)

    print(title)
    print(f"ACCURACY SCORE: {round(acc, 5)* 100}%")
    print(f"CLASSIFICATION REPORT: \n{class_report}")





