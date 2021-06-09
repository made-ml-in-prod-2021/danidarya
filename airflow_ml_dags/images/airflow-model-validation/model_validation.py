import os
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_percentage_error,r2_score
import click

@click.command("model_validation")
@click.option("--data-dir")
@click.option("--model-dir")
@click.option("--metrics-dir")
def validation(data_dir:str, model_dir: str, metrics_dir:str):
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "rb") as m:
        model =  pickle.load(m)
    data_val = pd.read_csv(os.path.join(data_dir, "data_val.csv"))
    y_val = data_val["target"].values
    X_val = data_val.drop("target", axis=1).values
    predictions = model.predict(X_val)
    mape = mean_absolute_percentage_error(y_val,predictions)
    r2 = r2_score(y_val,predictions)
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, "metrics.txt"), "w") as f:
        f.write(f"MAPE: {mape}, r2_score: {r2}")


if __name__ == '__main__':
    validation()