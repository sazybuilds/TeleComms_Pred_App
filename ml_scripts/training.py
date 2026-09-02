import os
from pathlib import Path
import logging

from sklearn.metrics import(
    accuracy_score,
    classification_report
)
import pandas as pd
import numpy as np

import sys
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# Ensure the root directory is in the python path to allow absolute imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ml_scripts import func as prep_func


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

# pyrefly: ignore [missing-import]
from joblib import dump

def train_model():
    try:
        load_dotenv()

        PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
        DATASET_PATH = PROJECT_ROOT / os.getenv("DATASET_NAME")
        MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
        LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR") / os.getenv("LOG_NAME")


        TARGET_COL = os.getenv("TARGET_COL")
        TEST_SIZE = float(os.getenv("TEST_SIZE"))
        RANDOM_STATE = int(os.getenv("RANDOM_STATE"))


        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(LOG_PATH)
            ]
        )

        logging.info("Configuration complete")
        logging.info("Training Script Started")

        df = pd.read_csv(DATASET_PATH)
        logging.info(f"Dataset loaded with shape {df.shape}")

        id_column: str = "customerID"

        df = df.drop(columns=[id_column])
        logging.info(f"ID column {id_column} dropped due to unimportance")
        


        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        y= y.map({"No": 0, "Yes": 1})
        logging.info(f"Target column {TARGET_COL} has been mapped to numeric values 0 and 1")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            random_state=RANDOM_STATE,
            stratify=y,
            test_size=TEST_SIZE
        )


        binary_features = []#Use a function
        nominal_features = []#OneHotEncoding
        numeric_features = []#scaling


        for col in X_train.columns.tolist():
            num_unique_values = X_train[col].nunique()
            if num_unique_values == 2:
                binary_features.append(col)
            elif num_unique_values > 2 and num_unique_values < 10:
                nominal_features.append(col)
            else:
                numeric_features.append(col)

        binary_features.remove("SeniorCitizen")#Already encoded


        logging.info("Lists of different features created")
        COL_WITH_ZEROS = ["TotalCharges"]


        zero_to_nan  = FunctionTransformer(
            prep_func.replace_zeros_with_nan_df,
            kw_args={"cols": COL_WITH_ZEROS},
            validate=False #Trust me bro
        )

        encode_binary = FunctionTransformer(
            prep_func.encode_binary_cols,
            kw_args={"cols": binary_features},
            validate=False
        )

        modify_str_int_column = FunctionTransformer(
            prep_func.modify_total_charges,
            kw_args={"cols": COL_WITH_ZEROS},
            validate=False
        )


        binary_transformer = Pipeline(
            steps = [
            ("encode", encode_binary),
            ]
        )

        nominal_transformer = Pipeline(
            steps = [
            ("oneHot", OneHotEncoder())
            ]
        )

        numerical_transformer = Pipeline(
            steps = [
            ("modify_odd_column", modify_str_int_column),
            ("zero_to_nan", zero_to_nan),
            ("impute", SimpleImputer(strategy="median")), 
            ("scale", StandardScaler())
            ]
        )

        preprocess = ColumnTransformer(
            transformers=[
                ("binary", binary_transformer, binary_features),
                ("nominal", nominal_transformer, nominal_features),
                ("numerical", numerical_transformer, numeric_features)
            ]
        )

        logging.info("Preprocessing structure created")


        best_model_pipeline = Pipeline(
            steps = [
                ("preprocess", preprocess),
                ("model", LogisticRegression(
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    C=1,
                    l1_ratio=0,
                    max_iter=1000,
                    solver='liblinear'
                ))
            ]
        )

        logging.info("Model Pipeline created")
        best_model_pipeline.fit(X_train, y_train)
        logging.info("Model training complete")

        model_predictions_train = best_model_pipeline.predict(X_train)
        model_predictions_test = best_model_pipeline.predict(X_test)

        logging.info(f"{prep_func.print_evaluation("MODEL TRAINING EVALUTION REPORT", y_train, model_predictions_train)}")

        logging.info("Model evaluation on training dataset visualized")

        logging.info("Testing Trained Model")
        logging.info(f"{prep_func.print_evaluation("MODEL TRAINING EVALUTION REPORT", y_test, model_predictions_test)}")
        logging.info("Model evaluation on testing dataset visualized")

        logging.info("Training script finished")

        logging.info(f"Saving model to {MODEL_PATH}")
        dump(best_model_pipeline, MODEL_PATH)
        logging.info(f"Model saved to {MODEL_PATH}")


    except Exception as e:
        print(f"Training Failed: {e}")
        logging.info(f"Training script failed: {e}")
        raise


if __name__ == "__main__":
    train_model()





