import os
import pandas as pd
import click
from sklearn.model_selection import train_test_split


@click.command("data_splitting")
@click.option("--data_for_split_dir")
def data_splitting(data_for_split_dir: str):

    data = pd.read_csv(os.path.join(data_for_split_dir, "data_preprocessed.csv"))
    data_train, data_val = train_test_split(data, test_size=0.25)
    data_train.to_csv(os.path.join(data_for_split_dir, "data_train.csv"), index=False)
    data_val.to_csv(os.path.join(data_for_split_dir, "data_val.csv"), index=False)


if __name__ == '__main__':
    data_splitting()