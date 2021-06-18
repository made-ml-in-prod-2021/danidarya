import os

import click
from sklearn.datasets import load_diabetes


@click.command("data_generation")
@click.argument("output_dir")
def generate(output_dir: str):
    data, target = load_diabetes(return_X_y=True, as_frame=True)
    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, "data.csv"),index=False)
    target.to_csv(os.path.join(output_dir, "target.csv"),index=False)

if __name__ == '__main__':
    generate()