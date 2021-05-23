import logging
import os
import pickle
from typing import List, Union, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, conlist
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)
model: Optional[Pipeline] = None
transformer: Optional[ColumnTransformer] = None
app = FastAPI()


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


class HeartPredictModel(BaseModel):
    data: List[conlist(Union[float, str, None], min_items=13, max_items=13)]
    features: List[str]


class Response(BaseModel):
    target: int


def make_predict( data: List, features: List[str],
                  model: Pipeline,
                  transformer: ColumnTransformer) -> List[Response]:
    data = pd.DataFrame(data, columns=features)
    transformed_data = transformer.transform(data)
    predicts = model.predict(transformed_data)
    return [
        Response(target=int(pred)) for pred in predicts
    ]


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_model_and_transformer():
    global model, transformer
    model_path = os.getenv("PATH_TO_MODEL", default="models/model.pkl")
    transformer_path = os.getenv("PATH_TO_TRANSFORMER", default="models/transformer.pkl")
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)
    if transformer_path is None:
        err = f"PATH_TO_TRANSFORMER {transformer_path} is None"
        logger.error(err)
        raise RuntimeError(err)

    model = load_object(model_path)
    transformer = load_object(transformer_path)


@app.get("/health")
def health() -> bool:
    return not (model is None)


@app.get("/predict/", response_model=List[Response])
def predict(request: HeartPredictModel):
    return make_predict(request.data, request.features, model, transformer)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
