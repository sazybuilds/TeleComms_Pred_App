import os
from pathlib import Path
import logging
import pandas as pd

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from joblib import load
import sys

# Ensure the root directory is in the python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the func module so joblib can find the custom functions during unpickling
from ml_scripts import func

# Map 'func' to 'ml_scripts.func' in sys.modules to maintain backward compatibility 
# with the model that was trained before the directory restructure
sys.modules['func'] = sys.modules['ml_scripts.func']

try:
    load_dotenv()

    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))

    MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
    LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR") / os.getenv("LOG_NAME")

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_PATH)
        ]
    )

    logging.info("Configuration complete")



    logging.info("Loading Trained Model")
    loaded_model = load(MODEL_PATH)
    logging.info("Model Loaded")


except Exception as e:
    print(f"Error occured: {e}")
    logging.info(f"Error: {e}")
    raise



def predict(input_data: dict):
    df = pd.DataFrame([input_data])
    predicted_value = loaded_model.predict(df)[0]
    logging.info("Model made a prediction")
    return predicted_value







        


