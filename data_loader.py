import pandas as pd
from ucimlrepo import fetch_ucirepo

def load_dataset(dataset_id):
    dataset = fetch_ucirepo(id=dataset_id)
    X = dataset.data.features
    y = dataset.data.targets
    # Convert target variable to numerical format
    if dataset_id == 2:  # Adult dataset
        y_encoded = (y.iloc[:, 0] == ' >50K').astype(int)
    else:
        y_encoded = y.iloc[:, 0].astype('category').cat.codes
    return X, y_encoded
