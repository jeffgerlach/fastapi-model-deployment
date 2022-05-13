import pytest
import pandas as pd
from starter.starter.ml.data import process_data
from starter.starter.ml.model import train_model
from starter.starter.train_model import cat_features


@pytest.fixture
def data():
    df = pd.read_csv("starter/data/census_cleaned.csv", nrows=50)
    return df


@pytest.fixture
def processed_data(data):
    X, y, _, _ = process_data(data, cat_features, label="salary")
    return X, y


@pytest.fixture
def trained_model(processed_data):
    X, y = processed_data
    return train_model(X, y)
