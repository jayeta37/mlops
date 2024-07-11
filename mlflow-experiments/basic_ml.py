import os
import numpy as np
import pandas as pd
import argparse
import mlflow
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from ucimlrepo import fetch_ucirepo
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(URL):
    try:
        df = pd.read_csv(URL, sep=",")
        X = df.drop(columns="quality")
        label_encoder = LabelEncoder()
        X["color"] = label_encoder.fit_transform(df["color"])
        y = df.quality
        logger.info("Data processing completed.")
        return X, y
    except Exception as e:
        logger.error(f"Error in loading file: {e}")
        raise e
    
def eval_function(actual, pred):
    rmse = mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    logger.info("Evaluation completed.")
    return rmse, mae, r2

def main(alpha, l1_ratio):
    X, y = load_data(URL = "https://archive.ics.uci.edu/static/public/186/data.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6, test_size=0.2)

    mlflow.set_experiment("ElasticNet-ML-Model-1")
    with mlflow.start_run():
        mlflow.log_params(params={'alpha': alpha, 'l1_ratio': l1_ratio})
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=1)
        logger.info("Model retrieved.")
        model.fit(X=X_train, y=y_train)
        y_pred = model.predict(X_test)
        rmse, mae, r2 = eval_function(y_test, y_pred)

        mlflow.log_metrics(metrics={
            'rmse': rmse,
            'mae': mae,
            'r2-score': r2
        })
        mlflow.sklearn.log_model(model, "trained_model")
        logger.info("Model logging completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", "-a", type=float, default=0.2)
    parser.add_argument("--l1_ratio", "-l1", type=float, default=0.3)
    parser_args = parser.parse_args()
    main(parser_args.alpha, parser_args.l1_ratio)