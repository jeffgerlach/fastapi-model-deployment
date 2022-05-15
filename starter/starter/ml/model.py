from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall,
    and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def compute_model_slice_metrics(model, X, y, categorical_feature):
    """
    Validates the trained machine learning model using precision, recall,
    and F1 on a single categorical feature, returned in a dict.

    Inputs
    ------
    model : sklearn.ensemble.RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    y : np.array
        Known labels, binarized.
    categorical_feature: str
        Predicted labels, binarized.
    Returns
    -------
    metrics : dict
    """
    predictions = inference(model, X)
    metrics = {}
    for cat_feature_option in set(categorical_feature):
        mask = categorical_feature == cat_feature_option
        precision, recall, f_beta = compute_model_metrics(y[mask],
                                                          predictions[mask])
        metrics[cat_feature_option] = {'precision': precision,
                                       'recall': recall, 'f_beta': f_beta,
                                       'count': sum(mask)}
    return metrics


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.ensemble.RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)
