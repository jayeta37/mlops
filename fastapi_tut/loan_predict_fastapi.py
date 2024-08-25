from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(PACKAGE_ROOT)
print(PACKAGE_ROOT)

