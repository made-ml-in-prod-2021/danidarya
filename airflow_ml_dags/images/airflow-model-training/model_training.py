import os
import click
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

@click.command("model_training")
@click.option("--input-dir")
@click.option("--output-dir")
def train(input_dir: str, output_dir: str):
    data_path = os.path.join(input_dir, "data_train.csv")
    data = pd.read_csv(data_path)
    y = data['target'].values
    X = data.drop('target',axis=1).values
    model = RandomForestRegressor()
    model.fit(X,y)
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, "model.pkl")
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    train()