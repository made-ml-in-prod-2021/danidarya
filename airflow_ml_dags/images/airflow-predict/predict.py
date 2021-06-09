import os
import pandas as pd
import click
import pickle
model_path = os.environ['MODEL_PATH']

@click.command("predict")
@click.option("--input-dir")
@click.option("--output-dir")
def predict(input_dir: str, output_dir):

    with open(model_path, "rb") as m:
        model = pickle.load(m)
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    preds = model.predict(data)
    data["predict"] = preds
    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, "predictions.csv"))


if __name__ == '__main__':
    predict()