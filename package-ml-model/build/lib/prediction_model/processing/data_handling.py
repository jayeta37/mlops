import os
import pandas as pd
import joblib #for serialization and deserialization
from prediction_model.config import config

def load_dataset(file_name):
    file_path = os.path.join(config.DATA_PATH, file_name)
    data = pd.read_csv(file_path)
    return data

def save_pipeline(pipeline_to_save):
    save_path = os.path.join(config.SAVED_MODEL_PATH, config.MODEL_NAME)
    joblib.dump(pipeline_to_save, save_path)
    print(f"Model has been saved as {config.MODEL_NAME}")

def load_pipeline():
    save_path = os.path.join(config.SAVED_MODEL_PATH, config.MODEL_NAME)
    model_loaded = joblib.load(save_path)
    print(f"{config.MODEL_NAME} has been loaded.")
    return model_loaded