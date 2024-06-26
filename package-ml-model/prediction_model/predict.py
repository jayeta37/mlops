from prediction_model.processing import data_handling
import pandas as pd
import numpy as np
import joblib
from prediction_model.config import config
from prediction_model.processing.data_handling import load_pipeline

classification_pipeline = load_pipeline(config.MODEL_NAME)

def generate_predictions(data_input):
    data = pd.DataFrame(data_input)
    prediction = classification_pipeline.predict(data[config.FEATURES])
    output = np.where(prediction == 1, 'Y', 'N')
    result = {"Predictions" : output}
    return result

if __name__=="__main__":
    generate_predictions()