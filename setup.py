import io
import os
from pathlib import Path

from setuptools import find_packages, setup

#METADATA
NAME = "prediction_model"
DESCRIPTION = "Loan Prediction Model"
URL = "https://github.com/jayeta37/mlops"
EMAIL = "jay37tawade@gmail.com"
AUTHOR = "Jayesh Tawade"
REQUIRES_PYTHON = ">=3.7.0"

pwd = os.path.abspath(os.path.dirname(__file__))

def list_requirements(fname = "requirements.txt"):
    with io.open(os.path.join(pwd, fname), encoding="utf-8") as f:
        return f.read().splitlines()
    
try:
    with io.open(os.path.join(pwd, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR/NAME
about = {}

with open(PACKAGE_DIR/'VERSION') as f:
    __version__ = f.read().strip()
    about['__version__'] = __version__

setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    maintainer=AUTHOR,
    url=URL,
    packages=find_packages(exclude=("tests")),
    package_data={"prediction_model": ["VERSION"]},
    install_requires=list_requirements(),
    include_package_data=True
)