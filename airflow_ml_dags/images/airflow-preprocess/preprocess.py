import os
import pandas as pd
import click


@click.command("predict")
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess(input_dir: str, output_dir):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))
    data['target'] = target['target']
    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, "data_preprocessed.csv"),index=False)


if __name__ == '__main__':
    preprocess()