import pandas as pd
from ucimlrepo import fetch_ucirepo

def load_dataset(dataset_id):
    dataset = fetch_ucirepo(id=dataset_id)
    X = dataset.data.features
    y = dataset.data.targets
    return X, y
