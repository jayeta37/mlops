# Loan Prediction Model

A machine learning model for predicting loan approvals. This project is part of the MLOps series and includes end-to-end processes for data handling, model training, prediction, and deployment.

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Directory_Structure](#directory_structure)

## Description

This project aims to predict loan approvals using various machine learning techniques. It covers the entire MLOps lifecycle including data preprocessing, model training, evaluation, and deployment.

## Installation

To install the package, you need to have Python 3.7 or above. You can install the package directly from GitHub using pip.

```sh
pip install git+https://github.com/jayeta37/mlops.git
```

## Directory_Structure

MLOPS/
├── build/
├── dist/
├── package-ml-model/
│   ├── build/
│   ├── dist/
│   ├── prediction_model/
│   │   ├── __pycache__/
│   │   ├── config/
│   │   │   ├── __pycache__/
│   │   │   ├── config.py
│   │   ├── datasets/
│   │   │   ├── test.csv
│   │   │   ├── train.csv
│   │   ├── processing/
│   │   │   ├── __pycache__/
│   │   │   ├── data_handling.py
│   │   │   ├── preprocessing.py
│   │   ├── trained_models/
│   │   │   ├── classification.pkl
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   ├── predict.py
│   │   ├── training_pipeline.py
│   │   ├── VERSION
│   │   ├── prediction_model.egg-info/
│   ├── tests/
│   │   ├── __pycache__/
│   │   ├── .pytest_cache/
│   │   ├── pytest.ini
│   │   ├── test_prediction.py
│   ├── MANIFEST.in
│   ├── README.md
│   ├── requirements.txt
│   ├── setup.py
├── .gitignore
├── Loan-Prediction.ipynb
└── requirements.txt
