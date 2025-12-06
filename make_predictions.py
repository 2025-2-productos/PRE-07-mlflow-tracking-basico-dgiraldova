"""Predicition script for the MLflow model.

This script loads a model from MLflow and makes predicitons on a dataset.

$ python3 make_predictions.py

"""

import mlflow
import pandas as pd

FILE_PATH = "data/winequality-red.csv"

df = pd.read_csv(FILE_PATH)
y = df["quality"]
x = df.drop(columns="quality")


## Debe verificarse el run_id del modelo que se quiere cargar
## El run_id se puede obtener desde la interfaz web de MLflow

logged_model = "runs:/2d5648a37f9b4b05a7de57e0802ce2c1/model"
loaded_model = mlflow.pyfunc.load_model(logged_model)
y = loaded_model.predict(x)

print(y)
