import numpy as np
import pandas as pd
import requests

if __name__ == "__main__":
    data = pd.read_csv("data/data_for_pred.csv")
    request_features = list(data.columns)
    N_REQUESTS = data.shape[0]
    for i in range(N_REQUESTS):
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
        ]
        print(request_data)
        response = requests.get(
            "http://0.0.0:8000/predict/",
            json={"data": [request_data], "features": request_features},
        )
        print(response.status_code)
        print(response.json())