import sklearn
import numpy as np
from starter.starter.ml.model import train_model, compute_model_metrics, inference


def test_train_model(processed_data):
    X, y = processed_data
    model = train_model(X, y)
    assert type(model) == sklearn.ensemble._forest.RandomForestClassifier


def test_compute_model_metrics(processed_data, trained_model):
    X, y = processed_data
    predictions = inference(trained_model, X)
    precision, recall, f_beta = compute_model_metrics(y, predictions)
    assert np.issubdtype(type(precision), float)
    assert np.issubdtype(type(recall), float)
    assert np.issubdtype(type(f_beta), float)


def test_inference(processed_data, trained_model):
    X, y = processed_data
    predictions = inference(trained_model, X)
    assert len(predictions) == 50
    assert len(predictions) == len(X)
    assert type(predictions) == np.ndarray
