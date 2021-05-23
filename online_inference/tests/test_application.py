import os
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from app import app, load_model_and_transformer

PATH_TO_MODEL="models/model.pkl"
PATH_TO_TRANSFORMER="models/transformer.pkl"

@pytest.fixture(scope="session")
def initialize():
    load_model_and_transformer()

@pytest.fixture(scope="session")
def data_test() -> pd.DataFrame:
    return pd.DataFrame({"age":63,"sex":1,"cp":3,"trestbps": 145,"chol": 233,"fbs":1, "restecg":0,	"thalach":150,
                         "exang":0,	"oldpeak":2.3,	"slope": 0,	"ca":0,	"thal":1},index=[0])

def test_entry_point():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == "it is entry point of our predictor"

def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() in [ True, False]

def test_predict(data_test):
    with TestClient(app) as client:
        features = data_test.columns.tolist()
        data = data_test.values.tolist()
        response = client.get("/predict/", json={"data": data, "features": features})
        assert response.status_code == 200
        assert response.json()[0]["target"] == 1